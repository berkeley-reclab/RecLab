pylint --rcfile=.pylintrc reclab -f parseable -r n --load-plugins pylint_quotes
pycodestyle reclab --max-line-length=100 --exclude=reclab/recommenders/autorec/autorec_lib,reclab/recommenders/cfnade/cfnade_lib,reclab/recommenders/llorma/llorma_lib
pydocstyle reclab --match-dir="^(?!autorec_lib|cfnade_lib|llorma_lib).*"
pylint --rcfile=.pylintrc tests -f parseable -r n --load-plugins pylint_quotes
pycodestyle tests --max-line-length=100
pydocstyle tests
pylint --rcfile=.pylintrc experiments -f parseable -r n --load-plugins pylint_quotes
pycodestyle experiments --max-line-length=100 --exclude=experiments/experiments
pydocstyle experiments --match-dir="^(?!experiments/experiments).*"
