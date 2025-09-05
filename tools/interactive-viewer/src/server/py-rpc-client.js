// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// py-rpc-client.js
// ---------------------------------------------------------------------------
// Node.js wrapper for communicating with a long-lived Python subprocess
// (`rpc_server.py`) using a lightweight JSON-RPC protocol over stdin/stdout.
//
// Features:
// • Spawns and maintains a persistent Python child process
// • Sends method calls with arguments over stdin and receives structured JSON responses over stdout
// • Tracks pending requests using unique IDs and Promises
// • Gracefully shuts down and cancels any outstanding calls

const { spawn } = require("child_process");

const readline = require("readline");

class PyRpcClient {
  constructor(pythonPath = "python") {
    // Spawn once, keep hot
    this.py = spawn(pythonPath, ["-u", "rpc_server.py"], {
      cwd: __dirname,
      stdio: ["pipe", "pipe", "inherit"],
    });

    // Map of pending { id → {resolve, reject} }
    this.pending = new Map();

    // Set up readline on stdout
    const rl = readline.createInterface({ input: this.py.stdout });
    rl.on("line", this._handleLine.bind(this));
    this._nextId = 1;
  }

  _handleLine(line) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (err) {
      console.error("Invalid JSON from Python:", line);
      return;
    }

    const { id, result, error } = msg;
    const entry = this.pending.get(id);
    if (!entry) {
      console.warn("Got response for unknown id:", id);
      return;
    }
    this.pending.delete(id);
    if (error) {
      const err = new Error(error.message);
      err.type = error.type;
      err.trace = error.trace;
      entry.reject(err);
    } else {
      entry.resolve(result);
    }
  }

  call(method, args = {}) {
    if (this.py.killed) {
      return Promise.reject(new Error("Python process has exited"));
    }

    const id = (this._nextId++).toString();
    const payload = JSON.stringify({ id, method, args }) + "\n";

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.py.stdin.write(payload, "utf8", (err) => {
        if (err) {
          this.pending.delete(id);
          return reject(err);
        }
      });
    });
  }

  // Graceful shutdown
  async end() {
    // Optionally wait for all pending to finish or reject
    for (let [id, { reject }] of this.pending) {
      reject(new Error("Shutdown: call cancelled"));
      this.pending.delete(id);
    }
    this.py.stdin.end();
    await new Promise((r) => this.py.once("close", r));
  }
}

module.exports = PyRpcClient;
