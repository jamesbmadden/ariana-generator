import * as tf from '@tensorflow/tfjs';

async function main () {
  const model = await tf.loadLayersModel('../model/model.json');
  console.log('model loaded');
}

main();