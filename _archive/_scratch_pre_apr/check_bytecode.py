"""Check the bytecode of the destination contract and see if it's a proxy."""
from web3 import Web3

DEST = "0x89404369C1D90145462e38BA479970a3e1e6736E"

w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/84842078b09946638c03157f83405213", request_kwargs={"timeout": 15}))
dest_cs = Web3.to_checksum_address(DEST)

# Get the raw bytecode
code = w3.eth.get_code(dest_cs)
print(f"Contract: {DEST}")
print(f"Code length: {len(code)} bytes")
print(f"Bytecode hex: {code.hex()}")

# Check if it's an EIP-1167 minimal proxy
# Standard pattern: 363d3d373d3d3d363d73 <20-byte address> 5af43d82803e903d91602b57fd5bf3
if len(code) == 45 and code[:10].hex().startswith("363d3d373d3d3d363d73"):
    impl = "0x" + code[10:30].hex()
    print(f"\nEIP-1167 Minimal Proxy -> Implementation: {impl}")
elif len(code) == 23:
    print(f"\nNot standard EIP-1167 (23 bytes, not 45)")
    # Could be a custom minimal proxy or something else
    # Try to decode common patterns
    print(f"First 4 bytes: {code[:4].hex()}")
    print(f"Disassembly hint:")
    # Manual opcode decode for small bytecodes
    opcodes = {
        0x36: "CALLDATASIZE", 0x3d: "RETURNDATASIZE", 0x37: "CALLDATACOPY",
        0x3e: "RETURNDATACOPY", 0x60: "PUSH1", 0x61: "PUSH2",
        0x73: "PUSH20", 0x5a: "GAS", 0xf4: "DELEGATECALL",
        0xf1: "CALL", 0xfd: "REVERT", 0xf3: "RETURN", 0x00: "STOP",
        0x80: "DUP1", 0x82: "DUP3", 0x90: "SWAP1", 0x91: "SWAP2",
        0x57: "JUMPI", 0x56: "JUMP", 0x5b: "JUMPDEST",
        0x52: "MSTORE", 0x51: "MLOAD", 0x54: "SLOAD", 0x55: "SSTORE",
    }
    i = 0
    while i < len(code):
        op = code[i]
        name = opcodes.get(op, f"0x{op:02x}")
        if 0x60 <= op <= 0x7f:  # PUSHn
            n = op - 0x5f
            data = code[i+1:i+1+n].hex()
            print(f"  {i:3d}: PUSH{n} {data}")
            i += 1 + n
        else:
            print(f"  {i:3d}: {name}")
            i += 1

# Also check all 5 TXs from our EOA
EOA = Web3.to_checksum_address("0x4BFC40EA4051f84E90eA0a25998578f6191Acad9")
print(f"\n--- Checking recent blocks for transactions from {EOA} ---")
# Get latest block
latest = w3.eth.block_number
print(f"Latest block: {latest}")
print(f"TX block: 24738735 (diff: {latest - 24738735} blocks)")
