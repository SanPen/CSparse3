#!/usr/bin/env bash
swig -c++ -python csr.i
swig -c++ -python csc.i
#swig -c++ -python coo.i
swig -c++ -python dia.i
swig -c++ -python bsr.i
swig -c++ -python csgraph.i