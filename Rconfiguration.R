# R configuration on Amarel
# from scratch

# currently on Amarel, need to get updated pcre2
# we can load this as a module
# module use /projects/community/modulefiles
# module load pcre2/10.35-gc563

# for packages, need java and gcc
# module load java/14.0.1
# module load gcc/5.4

# we will also need zlib for data.table
# wget https://zlib.net/zlib-1.2.11.tar.gz
# gunzip zlib-1.2.11.tar.gz
# tar -xvf zlib-1.2.11.tar
# cd zlib-1.2.11
# ./configure --prefix=$HOME/zlib-1.2.11
# make
# make install

# alternatively full install
# wget https://ftp.pcre.org/pub/pcre/pcre2-10.35.tar.gz
# gunzip pcre2-10.35.tar.gz
# tar -xvf pcre2-10.35.tar
# cd pcre2-10.35
# ./configure --prefix=$HOME/pcre2-10.35
# make
# install

# wget https://cran.r-project.org/src/base/R-4/R-4.0.3.tar.gz
# gunzip R-4.0.3.tar.gz
# tar -xvf R-4.0.3.tar
# 
# go into R directory to run
# cd R-4.0.3
# ./configure --prefix=$HOME/R-4.0.3
# make
# make install

# set paths
# export PATH=/home/rwomack/R-4.0.3/bin:$PATH
# export C_INCLUDE_PATH=/home/rwomack/R-4.0.3/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=/home/rwomack/R-4.0.3/include:$CPLUS_INCLUDE_PATH
# export LIBRARY_PATH=/home/rwomack/R-4.0.3/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/rwomack/R-4.0.3/lib:$LD_LIBRARY_PATH
# export MANPATH=/home/rwomack/R-4.0.3/share/man:$MANPATH

# and similarly for other installs

# tidyverse requires libsodium