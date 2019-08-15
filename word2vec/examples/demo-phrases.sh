#make
#if [ ! -e Training ]; then
#  wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/Training.gz
#  gzip -d Training.gz -f
#fi
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < Training | tr -c "A-Za-z'_ \n" " " > Training-norm0
time ./word2phrase -train Training-norm0 -output Training-norm0-phrase0 -threshold 200 -debug 2
time ./word2phrase -train Training-norm0-phrase0 -output Training-norm0-phrase1 -threshold 100 -debug 2
tr A-Z a-z < Training-norm0-phrase1 > Training-norm1-phrase1
time ./word2vec -train Training-norm1-phrase1 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
./distance vectors-phrase.bin
