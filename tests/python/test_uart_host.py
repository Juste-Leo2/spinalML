# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import struct
import pytest
from spinalml_cli.uart_host import (
    UartHost,
    CMD_CSR_WRITE,
    CMD_MEM_WRITE,
    CMD_READ_LOGITS,
    CMD_STATUS,
    CMD_VERSION,
    STATUS_BIT_AR_VALID,
    STATUS_BIT_R_VALID,
    STATUS_BIT_CSR_AW_VALID,
    STATUS_BIT_CSR_W_VALID,
    STATUS_BIT_OUT_VALID,
    STATUS_BIT_ACC_DONE,
    STATUS_BIT_ACC_BUSY,
    STATUS_BIT_PROTO,
)


class DummyUartHost(UartHost):
    def __init__(self):
        self.sent_bytes = bytearray()
        self.recv_queue = bytearray()

    def send(self, byte: int) -> None:
        self.sent_bytes.append(byte & 0xFF)

    def recv(self, timeout: float = 1.0) -> int:
        if not self.recv_queue:
            raise TimeoutError("No data available in dummy receive buffer")
        return self.recv_queue.pop(0)


def test_cmd_csr_write_encoding():
    host = DummyUartHost()
    addr = 0x00000008
    val = 0x00010000
    encoded = host.cmd_csr_write(addr, val)

    assert len(encoded) == 9
    assert encoded[0] == CMD_CSR_WRITE
    unpacked_addr, unpacked_val = struct.unpack("<II", encoded[1:9])
    assert unpacked_addr == addr
    assert unpacked_val == val


def test_cmd_mem_write_encoding():
    host = DummyUartHost()
    addr = 0x00020000
    data = bytes(range(16))  # 16 bytes = 2 words of 64-bit
    encoded = host.cmd_mem_write(addr, data)

    assert len(encoded) == 1 + 4 + 4 + len(data)
    assert encoded[0] == CMD_MEM_WRITE
    unpacked_addr, unpacked_len = struct.unpack("<II", encoded[1:9])
    assert unpacked_addr == addr
    assert unpacked_len == len(data)
    assert encoded[9:] == data


def test_cmd_mem_write_non_multiple_of_8():
    host = DummyUartHost()
    with pytest.raises(ValueError, match="multiple of 8"):
        host.cmd_mem_write(0x10000, bytes(7))


def test_cmd_read_logits():
    host = DummyUartHost()
    encoded = host.cmd_read_logits()
    assert encoded == bytes([CMD_READ_LOGITS])


def test_cmd_status():
    host = DummyUartHost()
    encoded = host.cmd_status()
    assert encoded == bytes([CMD_STATUS])


def test_cmd_version():
    host = DummyUartHost()
    encoded = host.cmd_version()
    assert encoded == bytes([CMD_VERSION])


def test_write_csr():
    host = DummyUartHost()
    host.write_csr(0x0C, 0x20000)
    expected = bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x0C, 0x20000)
    assert bytes(host.sent_bytes) == expected


def test_write_memory():
    host = DummyUartHost()
    payload = bytes(range(8))
    host.write_memory(0x10000, payload)
    expected = bytes([CMD_MEM_WRITE]) + struct.pack("<II", 0x10000, 8) + payload
    assert bytes(host.sent_bytes) == expected


def test_read_logits():
    host = DummyUartHost()
    host.recv_queue.extend([0x10, 0x20, 0x30])
    received = host.read_logits(count=3)
    assert bytes(host.sent_bytes) == bytes([CMD_READ_LOGITS])
    assert received == bytes([0x10, 0x20, 0x30])


def test_get_status():
    host = DummyUartHost()
    expected_status = (1 << STATUS_BIT_PROTO) | (1 << STATUS_BIT_ACC_DONE)
    host.recv_queue.append(expected_status)
    st = host.get_status()
    assert bytes(host.sent_bytes) == bytes([CMD_STATUS])
    assert st == expected_status


def test_version():
    host = DummyUartHost()
    host.recv_queue.append(0x01)
    v = host.version()
    assert bytes(host.sent_bytes) == bytes([CMD_VERSION])
    assert v == 0x01
