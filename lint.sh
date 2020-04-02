pylint --rcfile=.pylintrc reclab -f parseable -r n --load-plugins pylint_quotes
pycodestyle reclab --max-line-length=100 --exclude=reclab/recommenders/autorec/,reclab/recommenders/cfnade/cfnade_lib_keras
pydocstyle reclab --match-dir="^(?!autorec_lib|cfnade_lib_keras).*" 
pylint --rcfile=.pylintrc tests -f parseable -r n --load-plugins pylint_quotes
pycodestyle tests --max-line-length=100 
pydocstyle tests  