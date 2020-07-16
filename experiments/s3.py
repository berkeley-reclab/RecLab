"""A module to fetch and save experiment data from s3."""
import codecs
import datetime
import io
import json
import os
import pickle
import subprocess

import boto3

# The name of the file temporarily created for uploads to S3.
TEMP_FILE_NAME = 'temp.out'
# The id of all users that will have access to the S3 objects.
AWS_IDS = ['acde32a12806f031eb2518b0c2aca259ba031314143dfe2fab1bf6207af665f0',
           '4367fa3f02d247984d07c4e475f3e4e2abe5f95c6549f494f6784027cbb62601',
           '9fa447cf916c8ee7d02047335fdf9055c1627518e78b7168c6c737a403c3035f',
           '634b7a0686be3590c1808efc465ea9db660233386f1ad0bbbe3cabab19ae2564']
ID_STR = ','.join(['id=' + aws_id for aws_id in AWS_IDS])


def git_username():
    """Get the git username of the person running the code."""
    return subprocess.check_output(['git', 'config', 'user.name']).decode('ascii').strip()


def git_hash():
    """Get the current git hash of the code."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def git_branch():
    """Get the current git branch of the code."""
    return subprocess.check_output(['git', 'rev-parse',
                                    '--abbrev-ref', 'HEAD']).decode('ascii').strip()


def compute_across_trials(bucket_name,
                          data_dir,
                          env_name,
                          rec_names,
                          seeds,
                          func,
                          load_dense=False):
    """Apply func to all the trials of an experiment and return a list of func's return values.

    This function loads one trial at a time to prevent memory issues.

    """
    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member
    results = []
    for seed in seeds:
        all_ratings = []
        all_predictions = []
        for rec_name in rec_names:
            dir_name = experiment_dir_name(data_dir, env_name, rec_name, seed)
            (_, ratings, predictions,
             dense_ratings, dense_predictions, _) = load_trial(bucket,
                                                               dir_name,
                                                               load_dense=load_dense)
            if load_dense:
                ratings = dense_ratings
                predictions = dense_predictions
            all_ratings.append(ratings)
            all_predictions.append(predictions)
        results.append(func(ratings=all_ratings,
                            predictions=all_predictions))

        # Make these variables out of scope so they can be garbage collected.
        ratings = None
        predictions = None
        dense_ratings = None
        dense_predictions = None
        all_ratings = None
        all_predictions = None

    return results


def experiment_dir_name(data_dir, env_name, rec_name, trial_seed):
    """Get the directory name that corresponds to a given trial."""
    if data_dir is None:
        return None
    return os.path.join(data_dir, env_name, rec_name, 'trials', 'seed_' + str(trial_seed), '')


def dir_exists(bucket_name, dir_name):
    """Check if a directory exists in S3."""
    if bucket_name is None:
        return False

    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member

    # We can't use len here so do this instead.
    exists = False
    for _ in bucket.objects.filter(Prefix=dir_name):
        exists = True
        break

    return exists


def save_trial(bucket_name,
               dir_name,
               env_name,
               rec_name,
               rec_hyperparameters,
               ratings,
               predictions,
               dense_ratings,
               dense_predictions,
               recommendations,
               online_users,
               env_snapshots):
    """Save a trial in s3 within the given directory."""
    info = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'environment': env_name,
        'git branch': git_branch(),
        'git hash': git_hash(),
        'git username': git_username(),
        'recommender': rec_name,
    }
    serialize_and_put(bucket_name, dir_name, 'info', info, use_json=True)
    serialize_and_put(
        bucket_name, dir_name, 'rec_hyperparameters', rec_hyperparameters, use_json=True)
    serialize_and_put(bucket_name, dir_name, 'ratings', ratings)
    serialize_and_put(bucket_name, dir_name, 'predictions', predictions)
    serialize_and_put(bucket_name, dir_name, 'dense_ratings', dense_ratings)
    serialize_and_put(bucket_name, dir_name, 'dense_predictions', dense_predictions)
    serialize_and_put(bucket_name, dir_name, 'recommendations', recommendations)
    serialize_and_put(bucket_name, dir_name, 'online_users', online_users)
    serialize_and_put(bucket_name, dir_name, 'env_snapshots', env_snapshots)


def load_trial(bucket_name, dir_name, load_dense=True):
    """Load a trial saved in a given directory within S3."""
    ratings = get_and_unserialize(bucket_name, dir_name, 'ratings')
    predictions = get_and_unserialize(bucket_name, dir_name, 'predictions')
    rec_hyperparameters = get_and_unserialize(
        bucket_name, dir_name, 'rec_hyperparameters', use_json=True)
    if load_dense:
        dense_ratings = get_and_unserialize(bucket_name, dir_name, 'dense_ratings')
        dense_predictions = get_and_unserialize(bucket_name, dir_name, 'dense_predictions')
        env_snapshots = get_and_unserialize(bucket_name, dir_name, 'env_snapshots')
    else:
        dense_ratings = None
        dense_predictions = None
        env_snapshots = None

    return (rec_hyperparameters, ratings, predictions,
            dense_ratings, dense_predictions, env_snapshots)


def get_and_unserialize(bucket_name, dir_name, file_name, use_json=False):
    """Retrieve an object from S3 located at `dir_name/file_name`."""
    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member
    file_name = os.path.join(dir_name, file_name)
    if use_json:
        file_name = file_name + '.json'
    else:
        file_name = file_name + '.pickle'

    with open(TEMP_FILE_NAME, 'wb') as temp_file:
        bucket.download_fileobj(Key=file_name, Fileobj=temp_file)

    with open(TEMP_FILE_NAME, 'rb') as temp_file:
        if use_json:
            obj = json.load(temp_file)
        else:
            obj = pickle.load(temp_file)
    os.remove(TEMP_FILE_NAME)

    return obj


def serialize_and_put(bucket_name, dir_name, name, obj, use_json=False):
    """Serialize an object and upload it to S3."""
    bucket = boto3.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member
    file_name = os.path.join(dir_name, name)
    with open(TEMP_FILE_NAME, 'wb') as temp_file:
        if use_json:
            json.dump(obj, codecs.getwriter('utf-8')(temp_file),
                      sort_keys=True, indent=4)
            file_name = file_name + '.json'
        else:
            pickle.dump(obj, temp_file, protocol=4)
            file_name = file_name + '.pickle'

    with open(TEMP_FILE_NAME, 'rb') as temp_file:
        bucket.upload_fileobj(Key=file_name, Fileobj=temp_file,
                              ExtraArgs={'GrantFullControl': ID_STR})

    os.remove(TEMP_FILE_NAME)


def put_dataframe(bucket, dir_name, name, dataframe):
    """Upload a dataframe to S3 as a csv file."""
    with io.StringIO() as stream:
        dataframe.to_csv(stream)
        csv_str = stream.getvalue()

    with io.BytesIO() as stream:
        stream.write(csv_str.encode('utf-8'))
        stream.seek(0)
        file_name = os.path.join(dir_name, name + '.csv')
        bucket.upload_fileobj(Key=file_name, Fileobj=stream,
                              ExtraArgs={'GrantFullControl': ID_STR})
