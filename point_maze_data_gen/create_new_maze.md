
## Steps to create new maze environment and dataset

- Install `Gymnasium-Robotics`
  ```
  git clone https://github.com/Farama-Foundation/Gymnasium-Robotics.git
  cd Gymnasium-Robotics
  pip install -e .
  ```

- Create customized maze map
  Navigate to `Gymnasium-Robotics/gymnasium_robotics/envs/maze`, create new maze layout in `maps.py`.
- Regiest new environment
  Regester new environment in the file 'gymnasium_robotics/__init__.py`
  You can find my example code under the folder `./my_example`.

- Generate offline dataset
  Follow the example provided in 'pointmaze_XXlarge_data_gen.py'. The maze visualization code can be found in the corresponding Jupyter notebook.