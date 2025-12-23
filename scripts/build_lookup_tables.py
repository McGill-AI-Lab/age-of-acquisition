"""
There is no need to build lookup tables, so this file is purely meant
to test that src/lexical_features is working.

Instructions on how to set up are in src/lexical_features/README.md
"""

from lexical_features import *

print(f'aoa("hello") -> {aoa("hello")}')
print(f'aoa("dog") -> {aoa("dog")}')
print(f'aoa("unforgiving") -> {aoa("unforgiving")}')

print(f'conc("dog") -> {conc("dog")}')
print(f'conc("bite the bullet") -> {conc("bite the bullet")}')

print(f'freq("dog") -> {freq("dog")}') 
print(f'freq("unforgiving") -> {freq("unforgiving")}')

print(f'phon("dog") -> {phon("dog")}')
print(f'phon("worcestershire") -> {phon("worcestershire")}')