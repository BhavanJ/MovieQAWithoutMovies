# MovieQAWithoutMovies
Ranked 1st on [MovieQA leaderboard](http://movieqa.cs.toronto.edu/leaderboard/) on 4 out of 5 categories.


![Wiki](https://bhavanj.github.io/MovieQAWithoutMovies/images/Teaser_straight_arrows.jpg)

We have provided the processed data to run the code and get different results as mentioned in the paper. The accompanying paper can be found at the [project page](https://bhavanj.github.io/MovieQAWithoutMovies/). Follow the below steps to run our code.



## Setup the repository

1) Clone this repo - git clone https://github.com/BhavanJ/MovieQAWithoutMovies
2) Download the contents of data and w2v folders from [here](https://drive.google.com/drive/folders/16_GqxxY-w5Bz2yz4uQWupNqF43Wwishr?usp=sharing). These are the processed data.
3) Download the file GoogleNews-vectors-negative300.bin.gz from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and save it in w2v_model/Glove



## Train and replicate the results
```
python main.py

TODO: explain different settings
```

## Paper

Bhavan Jasani, Rohit Girdhar, Deva Ramanan, ["Are we asking the right questions in MovieQA?"](https://bhavanj.github.io/MovieQAWithoutMovies/), ICCVW, 2019.

```
@inproceedings{BJ_ICCV_2019,
  author    = {Bhavan Jasani, Rohit Girdhar and Deva Ramanan},
  title     = {Are we asking the right questions in MovieQA?},
  booktitle = {ICCVW},
  year      = {2019},
}
```


This code is built upon [Layered Memory Network](https://github.com/bowong/Layered-Memory-Network). We would like to thank them for providing some of their processed data.

