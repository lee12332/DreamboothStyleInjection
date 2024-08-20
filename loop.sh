#!/bin/bash

MAX_NUM=28

for ((number = 0; number < $MAX_NUM; number+=1)); do

      COMMAND="python run.py --sty_fn=./B --style_item=$number"     
      eval $COMMAND 
done