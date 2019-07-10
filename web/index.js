import * as tf from '@tensorflow/tfjs';

async function main () {
  // load the models
  const model = await tf.loadLayersModel('../model/model.json');

  // load the lyrics
  let response = await fetch('../python/lyrics.txt');
  const lyrics = await response.text();

  // Create a set of letters
  const vocab = Array.from(new Set(lyrics));

  const char2idx = char => vocab.indexOf(char);

  {
    // variables for generation
    let numGenerate = 2;
    let temperature = 1;
    let textGenerated = 1;
    let startString = '[Chorus';

    let inputEval = tf.expandDims(startString.split('').map(char2idx), 0);

    model.resetStates();
    for (let i = 0; i < numGenerate; i++) {
      let predictions = tf.squeeze(model.predict(inputEval), 0);
      let predictedID = tf.multinomial(predictions, 1);
    }
  }
}

main();