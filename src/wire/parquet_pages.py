# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import struct
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

# Thrift compact-protocol field types
_BOOL_TRUE, _BOOL_FALSE, _BYTE, _I16, _I32, _I64, _DOUBLE, _BINARY, _LIST, _SET, _MAP, _STRUCT = range(1, 13)

# parquet.thrift enums
DATA_PAGE, INDEX_PAGE, DICTIONARY_PAGE, DATA_PAGE_V2 = 0, 1, 2, 3
PLAIN, RLE = 0, 3


def _varint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def read_thrift_struct(buf, pos: int) -> tuple[dict[int, object], int]:
    """Decode one compact-protocol struct into {field id: value}; nested structs become dicts, lists become lists.

    Only what a parquet PageHeader needs is supported; anything else raises."""
    fields: dict[int, object] = {}
    last = 0
    while True:
        byte = buf[pos]
        pos += 1
        if byte == 0:
            return fields, pos
        delta, ctype = byte >> 4, byte & 0x0F
        if delta:
            fid = last + delta
        else:
            raw, pos = _varint(buf, pos)
            fid = _zigzag(raw)
        last = fid
        fields[fid], pos = _read_thrift_value(buf, pos, ctype)


def _read_thrift_value(buf, pos: int, ctype: int) -> tuple[object, int]:
    if ctype == _BOOL_TRUE:
        return True, pos
    if ctype == _BOOL_FALSE:
        return False, pos
    if ctype == _BYTE:
        return struct.unpack_from("b", buf, pos)[0], pos + 1
    if ctype in (_I16, _I32, _I64):
        raw, pos = _varint(buf, pos)
        return _zigzag(raw), pos
    if ctype == _DOUBLE:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if ctype == _BINARY:
        n, pos = _varint(buf, pos)
        return bytes(buf[pos:pos + n]), pos + n
    if ctype in (_LIST, _SET):
        head = buf[pos]
        pos += 1
        size, etype = head >> 4, head & 0x0F
        if size == 15:
            size, pos = _varint(buf, pos)
        items = []
        for _ in range(size):
            item, pos = _read_thrift_value(buf, pos, etype)
            items.append(item)
        return items, pos
    if ctype == _STRUCT:
        return read_thrift_struct(buf, pos)
    raise ValueError(f"unsupported thrift compact type {ctype}")


@dataclass(frozen=True)
class PageHeader:
    type: int
    uncompressed_size: int
    compressed_size: int
    num_values: int
    encoding: int
    header_size: int


def read_page_header(buf, pos: int = 0) -> PageHeader:
    """Parse the PageHeader that starts at `pos`; `header_size` tells where the page body begins."""
    fields, end = read_thrift_struct(buf, pos)
    page_type = fields[1]
    if page_type == DATA_PAGE:
        body = fields[5]
        num_values, encoding = body[1], body[2]
        if body[3] != RLE or body[4] != RLE:
            raise ValueError(f"levels encoded with {body[3]}/{body[4]}, only RLE is supported")
    elif page_type == DICTIONARY_PAGE:
        body = fields[7]
        num_values, encoding = body[1], body[2]
    elif page_type == DATA_PAGE_V2:
        raise ValueError("data page v2 is not supported")
    else:
        raise ValueError(f"unexpected page type {page_type}")
    return PageHeader(page_type, fields[2], fields[3], num_values, encoding, end - pos)


def decode_levels(data, bit_width: int, count: int) -> np.ndarray:
    """RLE / bit-packed hybrid decoder for repetition and definition levels (bit widths up to 8)."""
    out = np.empty(count, dtype=np.uint8)
    filled = pos = 0
    byte_width = (bit_width + 7) // 8
    while filled < count:
        header, pos = _varint(data, pos)
        if header & 1:
            groups = header >> 1
            nbytes = groups * bit_width
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=nbytes, offset=pos), bitorder="little")
            values = bits.reshape(-1, bit_width) @ (1 << np.arange(bit_width, dtype=np.uint8))
            pos += nbytes
        else:
            run = header >> 1
            value = int.from_bytes(data[pos:pos + byte_width], "little")
            pos += byte_width
            values = np.full(run, value, dtype=np.uint8)
        take = min(len(values), count - filled)
        out[filled:filled + take] = values[:take]
        filled += take
    return out


def decompress(body, codec: str, uncompressed_size: int) -> bytes:
    if codec.upper() == "UNCOMPRESSED":
        return bytes(body)
    return pa.Codec(codec.lower()).decompress(body, decompressed_size=uncompressed_size).to_pybytes()


@dataclass
class DecodedPage:
    rep: np.ndarray
    deff: np.ndarray
    values_offset: int
    body: bytes


def decode_page_levels(header: PageHeader, raw: bytes, codec: str, max_rep: int, max_def: int) -> DecodedPage:
    """Decompress a v1 data page and decode its levels; values stay raw (PLAIN byte arrays) until asked for."""
    if header.type != DATA_PAGE:
        raise ValueError(f"page type {header.type} where a data page was expected")
    if header.encoding != PLAIN:
        raise ValueError(f"values encoded with {header.encoding}, only PLAIN is supported")
    body = decompress(raw, codec, header.uncompressed_size)
    pos = 0
    if max_rep > 0:
        n = int.from_bytes(body[pos:pos + 4], "little")
        rep = decode_levels(body[pos + 4:pos + 4 + n], max_rep.bit_length(), header.num_values)
        pos += 4 + n
    else:
        rep = np.zeros(header.num_values, dtype=np.uint8)
    if max_def > 0:
        n = int.from_bytes(body[pos:pos + 4], "little")
        deff = decode_levels(body[pos + 4:pos + 4 + n], max_def.bit_length(), header.num_values)
        pos += 4 + n
    else:
        deff = np.zeros(header.num_values, dtype=np.uint8)
    return DecodedPage(rep, deff, pos, body)


def plain_byte_arrays(body: bytes, offset: int, count: int) -> list[bytes]:
    out = []
    pos = offset
    for _ in range(count):
        n = int.from_bytes(body[pos:pos + 4], "little")
        pos += 4
        out.append(body[pos:pos + n])
        pos += n
    return out


@dataclass(frozen=True)
class PageInfo:
    """Where a data page sits in the file and which rows it holds; `size` covers header and body."""

    offset: int
    size: int
    first_row: int
    num_rows: int
    continues: bool  # the page opens in the middle of a record started on the previous page


def scan_pages(chunk: bytes, chunk_offset: int, codec: str, max_rep: int, max_def: int) -> list[PageInfo]:
    """Walk every page of a column chunk (the bytes from data_page_offset for total_compressed_size) and count the
    records each page starts; this is the index that lets rows be fetched later without the chunk."""
    pages: list[PageInfo] = []
    pos = row = 0
    while pos < len(chunk):
        header = read_page_header(chunk, pos)
        raw = chunk[pos + header.header_size:pos + header.header_size + header.compressed_size]
        if header.type == DICTIONARY_PAGE:
            raise ValueError("dictionary-encoded column chunk, only PLAIN is supported")
        page = decode_page_levels(header, raw, codec, max_rep, max_def)
        starts = int(np.count_nonzero(page.rep == 0))
        continues = bool(len(page.rep) and page.rep[0] != 0)
        pages.append(PageInfo(chunk_offset + pos, header.header_size + header.compressed_size, row, starts, continues))
        row += starts
        pos += header.header_size + header.compressed_size
    return pages


def pages_for_row(pages: list[PageInfo], row: int) -> list[PageInfo]:
    """The page where `row` starts plus the pages it spills into.

    Only the last record started in a page can spill: it runs on through the physically following pages that
    `continue` it, ending in the first of them that starts a record of its own. `pages` may be a subset of the
    file's pages (the release ships only the needed ones), so a following page counts only if it is adjacent."""
    candidates = [p for p in pages if p.num_rows and p.first_row <= row < p.first_row + p.num_rows]
    if not candidates:
        raise KeyError(f"row {row} is not covered by the page table")
    start = pages.index(candidates[0])
    chosen = [pages[start]]
    if row == pages[start].first_row + pages[start].num_rows - 1:
        for p in pages[start + 1:]:
            if not p.continues or p.offset != chosen[-1].offset + chosen[-1].size:
                break
            chosen.append(p)
            if p.num_rows:
                break
    return chosen


def extract_rows(
    decoded: list[DecodedPage], first_row: int, rows: list[int], max_rep: int, max_def: int
) -> dict[int, list[str | None] | str | None]:
    """Values of `rows` out of consecutive decoded pages whose first complete record is `first_row`.

    A list<string> column (max_rep 1, max_def 3: optional list, optional element) yields lists; a plain optional
    string column (max_rep 0, max_def 1) yields strings."""
    rep = np.concatenate([p.rep for p in decoded])
    deff = np.concatenate([p.deff for p in decoded])
    values = []
    for p in decoded:
        values.extend(plain_byte_arrays(p.body, p.values_offset, int(np.count_nonzero(p.deff == max_def))))
    starts = np.flatnonzero(rep == 0)
    value_index = np.cumsum(deff == max_def) - (deff == max_def)
    out: dict[int, list[str | None] | str | None] = {}
    for row in rows:
        k = row - first_row
        if k < 0 or k >= len(starts):
            raise KeyError(f"row {row} is not in the fetched pages (rows {first_row}..{first_row + len(starts) - 1})")
        lo = int(starts[k])
        hi = int(starts[k + 1]) if k + 1 < len(starts) else len(rep)
        if max_rep == 0:
            out[row] = values[int(value_index[lo])].decode("utf-8") if deff[lo] == max_def else None
            continue
        if deff[lo] == 0:
            out[row] = None
        elif deff[lo] == 1:
            out[row] = []
        else:
            out[row] = [
                values[int(value_index[i])].decode("utf-8") if deff[i] == max_def else None for i in range(lo, hi)
            ]
    return out
