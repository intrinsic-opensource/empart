-----
# Interactive Mesh Simplification Front-End Tool

From the repo root,
```
conda activate empart
cd tools/interactive-viewer
export PORT=5004
npm start
```
By default the server listens on port 5005 and the front-end is hosted on port 5004. If you wanted to change the server port from `5005`, you need to change it in `.env` (front-end side) and `package.json` (server side).

### Port-Forwarding from Linux to Any Machine
Since the tool runs on Linux, you could forward its ports to your local machine (any platform) and access it in your browser:
```
ssh [the linux machine (i.e. joes@computer)] -L 5004:localhost:5004 -L 5005:localhost:5005
```
Here, the `-L <local>:<remote_host>:<remote>` flag forwards the local port to the remote port. 
Then, you can open your browser (`http://localhost:5004`)

### Resetting Stuck Ports
If you ned to restart the SSH connection or free up a port:
1. List processes using that port (e.g. 5004):
    ```bash
    lsof -i :5004
    ```
2. Kill the process (replace `<PID>` with the actul process ID)
    ```bash
    kill -0 <PID>
    ```

### ⚠ About This Code
This front-end code is part of ongoing research and experimentation. It's not optimized for production use, and some parts may priortize rapid iteration or exploratory workflows over production-level robustness.
You're welcome to use or adapt it, but expect a few rough edges.
