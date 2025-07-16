#!/bin/bash
CMD="cd ~/tatbot && git pull"
ssh rpi2 "$CMD"
ssh rpi1 "$CMD"
ssh trossen-ai "$CMD"
ssh ojo "$CMD"