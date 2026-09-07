# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""UART bridge L2 protocol (host side), see docs/uart_bridge.md.

This class only encodes/decodes the protocol: the actual byte transport is
provided by subclasses (serial port on Radxa/PC, or the cocotb simulator in
tests/python/test_uart*.py).
"""

import struct

CMD_CSR_WRITE = 0x43  # 'C'
CMD_MEM_WRITE = 0x57  # 'W'
CMD_READ_LOGITS = 0x52  # 'R'
CMD_STATUS = 0x53  # 'S'
CMD_VERSION = 0x56  # 'V'

STATUS_BIT_AR_VALID = 7
STATUS_BIT_R_VALID = 6
STATUS_BIT_CSR_AW_VALID = 5
STATUS_BIT_CSR_W_VALID = 4
STATUS_BIT_OUT_VALID = 3
STATUS_BIT_ACC_DONE = 2
STATUS_BIT_ACC_BUSY = 1
STATUS_BIT_PROTO = 0


class UartHost:
    """Encode/decode the L2 UART protocol.

    Wire order: command byte is 1 byte; 32-bit fields are LSB-first (per the
    reference RTL `{rx, acc[31:8]}` accumulation); 64-bit BRAM words are built
    LSB-first byte-per-word (first byte of a payload pair = least significant
    byte of the word).

    Subclasses provide `send(byte) -> None` and `recv(timeout) -> int` and
    the command helpers use them.
    """

    def send(self, byte: int) -> None:
        raise NotImplementedError

    def recv(self, timeout: float = 1.0) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Command encoders
    # ------------------------------------------------------------------
    def cmd_csr_write(self, addr: int, val: int) -> bytes:
        """'C' command: 4-byte addr (LSB-first per byte, reference RTL
        `{rx, acc[31:8]}` accumulation) + 4-byte val (LSB-first)."""
        return bytes([CMD_CSR_WRITE]) + struct.pack("<II", addr & 0xFFFFFFFF, val & 0xFFFFFFFF)

    def cmd_mem_write(self, addr: int, data: bytes) -> bytes:
        """'W' command: 4-byte addr (LSB-first) + 4-byte len + data (len % 8 == 0).

        The payload is written as-is: 8 bytes per 64-bit word, LSB-first byte
        per word (the first byte of the word is its least significant byte).
        """
        if len(data) % 8 != 0:
            raise ValueError(f"Mem write length must be a multiple of 8, got {len(data)}")
        return bytes([CMD_MEM_WRITE]) + struct.pack("<II", addr & 0xFFFFFFFF, len(data)) + data

    def cmd_read_logits(self) -> bytes:
        """'R' command (no payload)."""
        return bytes([CMD_READ_LOGITS])

    def cmd_status(self) -> bytes:
        """'S' command."""
        return bytes([CMD_STATUS])

    def cmd_version(self) -> bytes:
        """'V' command."""
        return bytes([CMD_VERSION])

    # ------------------------------------------------------------------
    # One-shot command senders (send payload then fetch the answer)
    # ------------------------------------------------------------------
    def write_csr(self, addr: int, val: int) -> None:
        for byte in self.cmd_csr_write(addr, val):
            self.send(byte)

    def write_memory(self, addr: int, data: bytes) -> None:
        for byte in self.cmd_mem_write(addr, data):
            self.send(byte)

    def read_logits(self, count: int, timeout: float = 1.0) -> bytes:
        """Send 'R' and return `count` logit bytes."""
        b = bytes([CMD_READ_LOGITS])
        for byte in b:
            self.send(byte)
        return bytes(self.recv(timeout) for _ in range(count))

    def get_status(self, timeout: float = 1.0) -> int:
        for byte in self.cmd_status():
            self.send(byte)
        return self.recv(timeout)

    def version(self, timeout: float = 1.0) -> int:
        for byte in self.cmd_version():
            self.send(byte)
        return self.recv(timeout)
