-----

# Empart: Interactive Mesh Simplification

This research project offers a web-based interactive tool for simplifying your meshes into convex shapes with varying error tolerances. 
The project is particularly aimed at finetuning collision geometry for physics simulation and robotics algorithms.

Read our pre-print paper explaining Empart on [arXiv](https://www.arxiv.org/pdf/2509.22847).

-----

## Getting Started

### Prerequisites
  * Ubuntu

Ensure you have these installed:
  * **Node.js**: Download from the [official Node.js website](https://nodejs.org/) or
    
    ```bash
    sudo apt install nodejs npm
    ```

  * **Conda (Anaconda or Miniconda)**: Download Miniconda from the [Conda website](https://docs.conda.io/en/latest/miniconda.html). or
    ```bash
    curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
    bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh
    ```
### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/intrinsic-opensource/empart.git
    cd empart
    ```
2.  **Set Up Python Environment with Conda:**
    ```bash
    conda create -n empart python=3.11
    conda activate empart
    pip install -r requirements.txt
    ```

3.  **Install Node.js Dependencies:**

    ```bash
    cd tools/interactive-viewer
    npm install
    ```

4. **Install Blender (optional for watertight mesh inputs):**    

    Note: If you are expecting to simplify meshes that are not watertight*, our tool requires blender to compute mesh intersections.
    Install blender from `https://www.blender.org/download/`
    ```
    tar -xf blender-4.5.1-linux-x64.tar.xz
    export PATH="/home/youruser/Downloads/blender-4.5.1-linux-x64:$PATH"
    source ~/.bashrc
    ```

    *A watertight triangle mesh is a closed 2-manifold or equivalently, every edge is shared by two faces.

-----

## Running the Application

```bash
cd tools/interactive-viewer
npm start
```

This command will start a local development server and open the application in your browser.

## Running Tests
Install playwright which is used to replay mouse and web events.
```bash
playwright install
```
Then, run the tests using pytest:
```bash
pytest
```
-----
## Citing Empart

Please include this bibtex entry when citing Empart:
```
@misc{vu2025empartinteractiveconvexdecomposition,
      title={Empart: Interactive Convex Decomposition for Converting Meshes to Parts}, 
      author={Brandon Vu and Shameek Ganguly and Pushkar Joshi},
      year={2025},
      eprint={2509.22847},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.22847}, 
}
```

-----

## Acknowledgements
`empart`'s ability to efficiently compute mesh intersections is made possible by amazing open source projects such as [Manifold3d](https://github.com/elalish/manifold) and [Blender](https://www.blender.org/)

