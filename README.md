# Code of G2-SCANN
We use dynamic link library, that is, **.so** file to protect our core code. So try to keep up with the following requirements as far as possible.
## Requirements
python=3.6.12

numpy=1.19.2

scikit-learn=0.23.2

matplotlib=3.3.2

scipy=1.5.2

networkx=2.5
## Usage
`cd G2-SCANN/`
### on synthetic datasets
`python main_synthetic.py --model Flame`

In synthetic_dataset repository, we provide 24 synthetic data files including A, B, Aggregation, S3, two_cluster, three_cluster, four_cluster, five_cluster, sn, fc1, Spiral2, Twomoons, Spiral3, Flame, E6, Jain, ThreeCircles, sk, line, circle, db4, db3, ls, and cth, where *.txt denotes raw datasets and *.gb0 indicates SPL adjacency matrix G_K that is pre-computed offline and then saved as a separate data file. Note that some *.gb0 files are temporarily missing here because the file sizes are too large to upload.
### on real datasets
`python main_real.py --model Iris`

We also provide 6 real dataset files including Iris, WDBC, Seeds, Libras_movement, Ionosphere, and Dermatology in real_datset repository.  
