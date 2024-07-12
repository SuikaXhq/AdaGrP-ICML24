python main.py --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic --generate
python main.py --method AdaGrP --early_stop --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic
python main.py --method FDiffS --early_stop --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic
python main.py --method IFCA --k -1 --early_stop --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic
python main.py --method FeSEM --k -1 --early_stop --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic
python main.py --method FedDrift --delta 60 --early_stop --M 10 --n 100 --S 3 --p 2 --p_equals_q --replicate_num 1 synthetic
