python main.py --method AdaGrP --M 344 --p 5 --q 3 --lr 0.005 --dataset noaa --type noaa --max_round 50 --early_stop
python main.py --method FedDrift --M 344 --p 5 --q 3 --lr 0.01 --delta 100 --dataset noaa --type noaa --max_round 50 --early_stop
python main.py --method FDiffS --M 344 --p 5 --q 3 --lr 0.005 --dataset noaa --type noaa --max_round 50 --early_stop