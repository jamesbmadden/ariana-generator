import * as tf from '@tensorflow/tfjs';

async function main () {
  const model = await tf.loadLayersModel('../model/model.json');
  console.log('model loaded');
  fetch('../python/lyrics.txt')
    .then(response => response.text())
    .then(text => console.log(text));
}

main();