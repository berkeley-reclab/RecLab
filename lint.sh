pylint --rcfile=.pylintrc reclab -f parseable -r n --load-plugins pylint_quotes
pycodestyle reclab --max-line-length=100 --exclude=reclab/recommenders/autorec/autorec_lib
pydocstyle reclab --match-dir="^(?!autorec_lib).*"
pylint --rcfile=.pylintrc tests -f parseable -r n --load-plugins pylint_quotes
pycodestyle tests --max-line-length=100
pydocstyle tests
pylint --rcfile=.pylintrc experiments -f parseable -r n --load-plugins pylint_quotes
pycodestyle experiments --max-line-length=100
pydocstyle experiments