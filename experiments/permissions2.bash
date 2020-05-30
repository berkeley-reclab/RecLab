for env in ml_100k
do
	for rec in TopPop RandomRec PerfectRec
	do
		for seed in 0 1 2 3 4 5 6 7 8 9
		do
			for filename in dense_predictions.pickle dense_ratings.pickle env_snapshots.pickle info.json online_users.pickle predictions.pickle ratings.pickle rec_hyperparameters.json recommendations.pickle
			do
				echo recsys-eval/master/$env/$rec/trials/seed_$seed/$filename
				aws s3api put-object-acl --acl bucket-owner-full-control --bucket recsys-eval --key master/$env/$rec/trials/seed_$seed/$filename
			done
		done
	done
done

