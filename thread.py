import threading
import logging
import time
import numpy as np

def func(*name):
    logging.info("Thread %s: starting", name[0])
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main      : before creating thread")
    arr=[]
    arr.append(2)
    arr.append(4)
    x = threading.Thread(target=func, args=(arr))
    logging.info("Main      : before running thread")
    x.start()
    logging.info("Main      : wait for the thread to finish")
    time.sleep(1)
    print('Hello world')
    x.join()
    logging.info("Main      : all done")