# Testing

To reproduce the paper’s experiment comparing V-HACD to our tool, run:

```bash
python testing/evaluate_convex_decompositions.py \
  testing/data/models/motor.glb \
  1   -0.501  0.228  -0.031  -0.339  0.281   0.031 \
  2   -0.182 -0.008   0.171  -0.148  0.062   0.215 \
  3    0.141 -0.010   0.177   0.227  0.064   0.220 \
  4    0.138 -0.013  -0.220   0.217  0.067  -0.182 \
  3   -0.195 -0.005  -0.217  -0.148  0.058  -0.175 \
  3    0.185  0.416  -0.243   0.238  0.495  -0.169 \
  3   -0.204  0.408   0.156  -0.115  0.501   0.243 \
  3 \
  --exp_name benchmark_vhacd \
  --method vhacd
```

- **`testing/data/models/motor.glb`**: input model  
- **`1 -0.501 …`**: sequence of decomposition parameters  
- **`--exp_name`**: name your experiment (here `benchmark_vhacd`)  
- **`--method`**: choose `vhacd` (or `coacd`)

Afterwards, you should be able to go into the `experiment_log` and see the experiment directory where there is a `result.json` storing the data and also other files for visualization. To view the .html files you can run `google-chrome [file].html`.

If you want to visualize the plots from the experiment run this:
```
python testing/create_plot.py experiments_log/benchmark_vhacd/result.json  --output result_plot
```

---

# Adding End-To-End (E2E) Tests

1. **Start the app**  
   ```bash
   npm start
   ```

2. **Open the E2E interface**  
   Navigate to:  
   ```
   https://localhost:5004/?e2e
   ```

3. **Interact & collect logs**  
   - Add boxes, process them, etc.  
   - Click **Copy E2E Logs** to save a `.log` file.

4. **Generate the Playwright test**  
   ```bash
   python testing/write_e2e_test.py path/to/your.log \
     > tests/my_new_test.py
   ```
   > This writes a new Playwright test into `tests/my_new_test.py`.  
   > Inspect or tweak it as needed.

5. **Run your test with pytest**  
   ```bash
   pytest tests/my_new_test.py
   ```

   - **Show console output**:  
     ```bash
     pytest tests/my_new_test.py --capture=tee-sys
     ```
   - **Run in headed mode** (slower, but visible in browser):  
     ```bash
     pytest tests/my_new_test.py --headed
     ```
