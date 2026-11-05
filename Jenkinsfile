pipeline {
    agent any

    parameters {
        choice(
            name: 'EXPERIMENT',
            choices: [
                'catboost',
                'catboost_balanced',
                'decision_tree',
                'random_forest',
                'logistic_regression',
                'logistic_regression_balanced',
                'all'
            ],
            description: 'Experiment from src/config.py'
        )
        string(name: 'CV_FOLDS', defaultValue: '5', description: 'Cross-validation folds (0 = no CV)')
    }

    environment {
        IMAGE = 'smart-irrigation-mlops:jenkins'
        DATA_VOLUME = 'smart-irrigation-data'
        CONTAINER = "train-${env.BUILD_NUMBER}"
        TRAIN_EXPERIMENT = "${params.EXPERIMENT ?: 'catboost'}"
        TRAIN_CV_FOLDS = "${params.CV_FOLDS ?: '5'}"
    }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build environment') {
            steps {
                sh 'docker build -t $IMAGE .'
            }
        }

        stage('Validate data') {
            steps {
                sh 'docker run --rm -v $DATA_VOLUME:/app/data $IMAGE python -m src.data'
            }
        }

        stage('Train') {
            steps {
                sh '''
                    docker run --name $CONTAINER -v $DATA_VOLUME:/app/data $IMAGE \
                        python -m src.train --experiment $TRAIN_EXPERIMENT --cv-folds $TRAIN_CV_FOLDS
                '''
            }
        }

        stage('Metrics') {
            steps {
                sh 'rm -rf models && docker cp $CONTAINER:/app/models ./models'
                sh 'for f in models/*/metrics.json; do echo "== $f"; cat "$f"; echo; done'
                archiveArtifacts artifacts: 'models/**/metrics.json, models/**/confusion_matrix.png',
                                 fingerprint: true
            }
        }
    }

    post {
        always {
            sh 'docker rm -f $CONTAINER || true'
        }
    }
}
