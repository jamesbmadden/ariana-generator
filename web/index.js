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
    let numGenerate = 100;
    let temperature = 1;
    let startString = '[Chorus';
    let generatedText = startString;

    let inputEval = tf.expandDims(startString.split('').map(char2idx), 0);

    model.resetStates();

    for (let i = 0; i < numGenerate; i++) {
      let predictions = model.predict(inputEval);

      predictions = tf.squeeze(predictions);

      const predictedId = await tf.multinomial(predictions, 1, null, false).data();
      console.log(predictedId[0]);;

      generatedText += vocab[predictedId[0]];
      predictions.dispose();
    }
    console.log(generatedText);
    document.body.textContent = generatedText;
  }
}

main();