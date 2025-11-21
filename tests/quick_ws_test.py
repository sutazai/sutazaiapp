import websockets
import asyncio
import json

async def test():
    ws = await websockets.connect('ws://localhost:11100/ws/test-123')
    msg = await ws.recv()
    print(json.dumps(json.loads(msg), indent=2))
    await ws.close()

asyncio.run(test())
