# Gallery Differ

## What it is about

A proof of concept that computes the difference between 2 photo galleries based on multiple perceptual hashing algorhtms. We define an `attack` as an attempt to pass a slightly changed version of a picture as a different picture entirely, and out purpose is to identify these kinds of attacks and mark the pictures as similar.

## Usage

`time python src/matching/index.py "/code/samples/002 - gin/original" "/code/samples/002 - gin/attack001"`
