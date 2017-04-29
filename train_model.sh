python3 model.py --model . --data data/ --epochs 50 2>&1 | tee train_model.txt && \
  git add train_model.txt && \
  cp weights.hdf5 model.h5 && \
  git commit -a -m "trained new model" &&
  python3 model.py --model . --data data/ --epochs 50 --load model.json 2>&1 | tee refine_model.txt && \
  git add refine_model.txt && \
  cp weights.hdf5 model.h5 && \
  git commit -a -m "trained new model" && sudo halt

