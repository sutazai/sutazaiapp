import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';

function frame(json: any): Buffer {
  const body = Buffer.from(JSON.stringify(json), 'utf8');
  const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'utf8');
  return Buffer.concat([header, body]);
}

async function readFrame(stream: NodeJS.ReadableStream, timeoutMs = 15000): Promise<any> {
  return new Promise((resolve, reject) => {
    let timer: NodeJS.Timeout | undefined;
    let buf = Buffer.alloc(0);
    const onData = (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk]);
      const idx = buf.indexOf('\r\n\r\n');
      if (idx !== -1) {
        const headerPart = buf.slice(0, idx).toString('utf8');
        const m = headerPart.match(/Content-Length:\s*(\d+)/i);
        if (!m) return;
        const len = parseInt(m[1], 10);
        const start = idx + 4;
        if (buf.length >= start + len) {
          const body = buf.slice(start, start + len).toString('utf8');
          stream.off('data', onData);
          if (timer) clearTimeout(timer);
          try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
        }
      }
    };
    stream.on('data', onData);
    timer = setTimeout(() => {
      stream.off('data', onData);
      reject(new Error('Timed out waiting for MCP frame'));
    }, timeoutMs);
  });
}

test('MCP stdio roundtrip: initialize, tools/list, tools/call', async () => {
  // Ensure docker is available; the tests image installs docker-cli and mounts the socket via compose
  const proc = spawn('docker', ['run', '--rm', '-i', 'mcp/sequentialthinking'], { stdio: ['pipe', 'pipe', 'pipe'] });

  // 1) initialize
  proc.stdin.write(frame({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      capabilities: { experimental: {} },
      clientInfo: { name: 'sutazai-playwright', version: '1.0.0' }
    }
  }));
  const initResp = await readFrame(proc.stdout);
  expect(initResp.jsonrpc).toBe('2.0');
  expect(initResp.id).toBe(1);
  expect(initResp.result).toBeTruthy();

  // 2) tools/list
  proc.stdin.write(frame({ jsonrpc: '2.0', id: 2, method: 'tools/list' }));
  const listResp = await readFrame(proc.stdout);
  expect(listResp.jsonrpc).toBe('2.0');
  expect(listResp.id).toBe(2);
  expect(Array.isArray(listResp.result?.tools)).toBeTruthy();
  const hasSeq = (listResp.result.tools as any[]).some(t => t?.name === 'sequentialthinking');
  expect(hasSeq).toBeTruthy();

  // 3) tools/call
  proc.stdin.write(frame({
    jsonrpc: '2.0',
    id: 3,
    method: 'tools/call',
    params: {
      name: 'sequentialthinking',
      arguments: {
        thought: 'Test',
        nextThoughtNeeded: false,
        thoughtNumber: 1,
        totalThoughts: 1
      }
    }
  }));
  const callResp = await readFrame(proc.stdout);
  expect(callResp.jsonrpc).toBe('2.0');
  expect(callResp.id).toBe(3);
  expect(Array.isArray(callResp.result?.content)).toBeTruthy();

  proc.stdin.end();
  proc.kill();
});

