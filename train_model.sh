source activate theano3.5
python3 model.py --model . --data car/ --epochs 150 2>&1 | tee train_model.txt && \
  git add train_model.txt && \
  #cp weights.hdf5 model.h5 && \
  #git commit -a -m "trained new model" &&
  #python3 model.py --model . --data car3/ --epochs 150 --load model.json 2>&1 | tee car3.txt && \
  #git add car3*.txt && \
  #cp weights.hdf5 model.h5 && \
  #git commit -a -m "trained new model with car3 data" && git push \ 
  #python3 model.py --model . --data car/ --epochs 150 --load model.json 2>&1 | tee car_1.txt && \
  #git add car_*.txt && \
  cp weights.hdf5 model.h5 && \
  git commit -a -m "trained new model with car data" && git push && \
  /usr/bin/rsync -ar --exclude-from='/home/ubuntu/exclude_me.txt' /home/ubuntu/CarND-CarBehaviourCloning /mnt/s3 && \
  sudo halt


