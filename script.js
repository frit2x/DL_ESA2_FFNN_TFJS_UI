const dataset = {
  x: [],
  y: [],
  xTrain: [],
  yTrain: [],
  xTest: [],
  yTest: [],
  yTrainNoisy: [],
  yTestNoisy: []
};

const models = {};

function f(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

function seededRandom(seed) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), 1 | r);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function randomUniformWithRand(min, max, rand) {
  return rand() * (max - min) + min;
}

function gaussianNoise(std, rand = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
}

function setStatus(text) {
  const status = document.getElementById('status');
  if (status) status.textContent = text;
}

function makeDataset(N = 100, noiseVar = 0.05, seed = null) {
  const rand = seed != null ? seededRandom(seed) : Math.random;
  const xs = Array.from({ length: N }, () => randomUniformWithRand(-2, 2, rand));
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }

  const ys = xs.map(f);
  const half = Math.floor(N / 2);

  dataset.x = xs;
  dataset.y = ys;
  dataset.xTrain = xs.slice(0, half);
  dataset.yTrain = ys.slice(0, half);
  dataset.xTest = xs.slice(half, half * 2);
  dataset.yTest = ys.slice(half, half * 2);

  const std = Math.sqrt(noiseVar);
  dataset.yTrainNoisy = dataset.yTrain.map((y) => y + gaussianNoise(std, rand));
  dataset.yTestNoisy = dataset.yTest.map((y) => y + gaussianNoise(std, rand));

  resetResults();
  plotDataset();
  setStatus(`Datensatz erzeugt: N=${N}, Rauschen V=${noiseVar.toFixed(3)} — alte Ergebnisse zurückgesetzt`);
}

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

function mse(yTrue, yPred) {
  let error = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const diff = yTrue[i] - yPred[i];
    error += diff * diff;
  }
  return error / yTrue.length;
}

function linspace(min, max, count) {
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

function plotDataset() {
  const layout = {
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  };

  // Clean dataset (no noise)
  Plotly.newPlot('plot-data-clean', [
    { x: dataset.xTrain, y: dataset.yTrain, mode: 'markers', name: 'Train ohne Rauschen', marker: { color: '#0a57ff', size: 8 } },
    { x: dataset.xTest, y: dataset.yTest, mode: 'markers', name: 'Test ohne Rauschen', marker: { color: '#4ea5ff', size: 8 } }
  ], Object.assign({ title: 'Datensatz ohne Rauschen' }, layout), { responsive: true });

  // Noisy dataset
  Plotly.newPlot('plot-data-noisy', [
    { x: dataset.xTrain, y: dataset.yTrainNoisy, mode: 'markers', name: 'Train mit Rauschen', marker: { color: '#ff4e4e', size: 8 } },
    { x: dataset.xTest, y: dataset.yTestNoisy, mode: 'markers', name: 'Test mit Rauschen', marker: { color: '#ff9a4e', size: 8 } }
  ], Object.assign({ title: 'Datensatz mit Rauschen' }, layout), { responsive: true });
}

function resetResults() {
  const plotIds = [
    'plot-clean-train', 'plot-clean-test',
    'plot-best-train', 'plot-best-test',
    'plot-over-train', 'plot-over-test'
  ];
  plotIds.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      Plotly.purge(element);
      element.innerHTML = '';
    }
  });

  const lossIds = [
    'loss-clean-train', 'loss-clean-test',
    'loss-best-train', 'loss-best-test',
    'loss-over-train', 'loss-over-test'
  ];
  lossIds.forEach((id) => {
    const element = document.getElementById(id);
    if (element) element.textContent = '';
  });

  const lossPlot = document.getElementById('plot-losses');
  if (lossPlot) {
    Plotly.purge(lossPlot);
    lossPlot.innerHTML = '';
  }

  Object.keys(models).forEach((key) => delete models[key]);
  lossHistory.clean = [];
  lossHistory.best = [];
  lossHistory.over = [];
}

function plotPredictionSingle(plotId, modelName, xData, yData, title, dataType = 'train', noisy = false) {
  const xGrid = linspace(-2, 2, 250);
  const predictions = predictModel(modelName, xGrid);

  // Choose marker color depending on train/test and noisy flag
  let markerColor = '#0a57ff';
  if (noisy) markerColor = dataType === 'train' ? '#ff6f6f' : '#ffb46f';
  else markerColor = dataType === 'train' ? '#0a57ff' : '#4ea5ff';

  const traces = [
    { x: xGrid, y: predictions, mode: 'lines', name: 'Modellkurve', line: { color: '#2dbe8f', width: 3 } },
    { x: xData, y: yData, mode: 'markers', name: dataType === 'train' ? 'Train' : 'Test', marker: { color: markerColor, size: 8 } }
  ];

  Plotly.newPlot(plotId, traces, {
    title,
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  }, { responsive: true });
}

function predictModel(name, xValues) {
  const model = models[name];
  if (!model) return xValues.map(() => 0);
  const tensor = tf.tensor2d(xValues, [xValues.length, 1]);
  const preds = model.predict(tensor);
  const values = Array.from(preds.dataSync());
  tensor.dispose();
  preds.dispose();
  return values;
}

async function train(name, xData, yData, epochs = 50) {
  const model = createModel();
  models[name] = model;

  const xs = tf.tensor2d(xData, [xData.length, 1]);
  const ys = tf.tensor2d(yData, [yData.length, 1]);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 32,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        addLossHistory(name, logs.loss, epoch + 1);
        await tf.nextFrame();
      }
    }
  });

  xs.dispose();
  ys.dispose();
  return history;
}

const lossHistory = { clean: [], best: [], over: [] };

function addLossHistory(name, loss, epoch) {
  lossHistory[name].push({ epoch, loss });
  plotLossHistory();
}

function plotLossHistory() {
  const traces = Object.entries(lossHistory)
    .filter(([, values]) => values.length > 0)
    .map(([name, values]) => ({
      x: values.map((item) => item.epoch),
      y: values.map((item) => item.loss),
      mode: 'lines+markers',
      name,
      marker: { size: 6 }
    }));

  Plotly.newPlot('plot-losses', traces, {
    title: 'Training Loss-Verlauf (MSE)',
    xaxis: { title: 'Epoche' },
    yaxis: { title: 'MSE' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  }, { responsive: true });
}

function evaluate(modelName, xData, yData) {
  const yPred = predictModel(modelName, xData);
  return mse(yData, yPred);
}

function getSavedModelAliasMap() {
  try {
    const raw = localStorage.getItem('tfjsModelAliasMap');
    return raw ? JSON.parse(raw) : {};
  } catch (error) {
    return {};
  }
}

function setSavedModelAlias(alias, modelKey) {
  const map = getSavedModelAliasMap();
  map[alias] = modelKey;
  localStorage.setItem('tfjsModelAliasMap', JSON.stringify(map));
}

function resolveSavedModelKey(alias) {
  const map = getSavedModelAliasMap();
  if (map[alias]) return map[alias];

  const normalized = alias.toLowerCase();
  if (normalized.includes('clean')) return 'clean';
  if (normalized.includes('best')) return 'best';
  if (normalized.includes('over')) return 'over';

  return ['clean', 'best', 'over'].includes(alias) ? alias : null;
}

function renderLoadedModel(alias, modelKey) {
  if (!dataset.xTrain.length) {
    setStatus('Modell geladen: Bitte zuerst Datensatz erzeugen oder laden.');
    return;
  }

  const key = modelKey || alias;
  const availableTrainPlot = document.getElementById(`plot-${key}-train`);
  const availableTestPlot = document.getElementById(`plot-${key}-test`);
  const availableTrainLoss = document.getElementById(`loss-${key}-train`);
  const availableTestLoss = document.getElementById(`loss-${key}-test`);
  if (!availableTrainPlot || !availableTestPlot || !availableTrainLoss || !availableTestLoss) {
    setStatus(`Modell geladen: ${alias}. Kein automatisches Update möglich für Modelltyp ${key}.`);
    return;
  }

  const isNoisy = key !== 'clean';
  const yTrain = isNoisy ? dataset.yTrainNoisy : dataset.yTrain;
  const yTest = isNoisy ? dataset.yTestNoisy : dataset.yTest;
  const label = key === 'clean' ? 'Clean' : key === 'best' ? 'Best-Fit' : key === 'over' ? 'Overfit' : 'Geladenes Modell';
  const plotType = isNoisy ? 'verrauscht' : 'unverrauscht';

  plotPredictionSingle(`plot-${key}-train`, alias, dataset.xTrain, yTrain, `${label} — Train (${plotType})`, 'train', isNoisy);
  plotPredictionSingle(`plot-${key}-test`, alias, dataset.xTest, yTest, `${label} — Test (${plotType})`, 'test', isNoisy);

  const trainLoss = evaluate(alias, dataset.xTrain, yTrain);
  const testLoss = evaluate(alias, dataset.xTest, yTest);
  availableTrainLoss.textContent = `Train MSE: ${trainLoss.toExponential(3)}`;
  availableTestLoss.textContent = `Test MSE: ${testLoss.toExponential(3)}`;
  setStatus(`Modell geladen und Visualisierung aktualisiert: ${alias} (${key})`);
}

function renderModelResults(key, yTrain, yTest, noisy) {
  const trainLoss = evaluate(key, dataset.xTrain, yTrain);
  const testLoss = evaluate(key, dataset.xTest, yTest);

  plotPredictionSingle(`plot-${key}-train`, key, dataset.xTrain, yTrain, `${key === 'clean' ? 'Unverrauscht' : key === 'best' ? 'Best-Fit' : 'Overfit'} — Train`, 'train', noisy);
  plotPredictionSingle(`plot-${key}-test`, key, dataset.xTest, yTest, `${key === 'clean' ? 'Unverrauscht' : key === 'best' ? 'Best-Fit' : 'Overfit'} — Test`, 'test', noisy);

  document.getElementById(`loss-${key}-train`).textContent = `Train MSE: ${trainLoss.toExponential(3)}`;
  document.getElementById(`loss-${key}-test`).textContent = `Test MSE: ${testLoss.toExponential(3)}`;
}

const DEFAULT_EPOCHS_CLEAN = 100;
const DEFAULT_EPOCHS_BEST = 350;
const DEFAULT_EPOCHS_OVER = 1250;

async function initialTraining() {
  setStatus('Initialisiere festen Datensatz und trainiere Modelle...');
  makeDataset(100, 0.05, 12345);

  const epochsClean = Number.parseInt(document.getElementById('epochsClean').value, 10) || DEFAULT_EPOCHS_CLEAN;
  const epochsBest = Number.parseInt(document.getElementById('epochsBest').value, 10) || DEFAULT_EPOCHS_BEST;
  const epochsOver = Number.parseInt(document.getElementById('epochsOver').value, 10) || DEFAULT_EPOCHS_OVER;

  await train('clean', dataset.xTrain, dataset.yTrain, epochsClean);
  renderModelResults('clean', dataset.yTrain, dataset.yTest, false);

  await train('best', dataset.xTrain, dataset.yTrainNoisy, epochsBest);
  renderModelResults('best', dataset.yTrainNoisy, dataset.yTestNoisy, true);

  await train('over', dataset.xTrain, dataset.yTrainNoisy, epochsOver);
  renderModelResults('over', dataset.yTrainNoisy, dataset.yTestNoisy, true);

  setStatus(`Initialisierung abgeschlossen. Fester Datensatz geladen und Modelle trainiert (Clean ${epochsClean}, Best-Fit ${epochsBest}, Overfit ${epochsOver}).`);
}

function downloadFile(data, filename, mimeType) {
  const blob = new Blob([data], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function setupEventHandlers() {
  document.getElementById('btn-generate').addEventListener('click', () => {
    const N = parseInt(document.getElementById('numPoints').value, 10);
    const V = parseFloat(document.getElementById('noiseVar').value);
    makeDataset(N, V);
  });

  document.getElementById('btn-save-data').addEventListener('click', () => {
    const data = JSON.stringify(dataset, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'dataset.json';
    link.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('btn-load-data').addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (event) => {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const data = JSON.parse(reader.result);
        Object.assign(dataset, data);
        resetResults();
        plotDataset();
        setStatus('Dataset geladen — alte Ergebnisse zurückgesetzt.');
      };
      reader.readAsText(file);
    };
    input.click();
  });

  document.getElementById('btn-save-model').addEventListener('click', async () => {
    const trainedNames = Object.keys(models);
    if (!trainedNames.length) {
      alert('Keine trainierten Modelle vorhanden. Bitte zuerst Trainieren.');
      return;
    }

    const modelKey = prompt('Welches trainierte Modell speichern? (clean, best, over)', trainedNames.includes('best') ? 'best' : trainedNames[0]);
    if (!modelKey) return;
    if (!models[modelKey]) {
      alert('Kein trainiertes Modell namens ' + modelKey + '. Verfügbare Modelle: ' + trainedNames.join(', '));
      return;
    }

    const alias = prompt('Speichere dieses Modell unter dem Alias:', modelKey);
    if (!alias) return;

    await models[modelKey].save('indexeddb://' + alias);
    setSavedModelAlias(alias, modelKey);
    setStatus(`Modell ${modelKey} gespeichert als ${alias}`);
  });

  document.getElementById('btn-load-model').addEventListener('click', async () => {
    const listed = await tf.io.listModels();
    const known = Object.keys(listed)
      .filter((key) => key.startsWith('indexeddb://'))
      .map((key) => key.replace('indexeddb://', ''));

    const defaultName = known.includes('best') ? 'best' : (known[0] || '');
    const name = prompt('Lade Modell aus IndexedDB. Verfügbare: ' + (known.length ? known.join(', ') : 'keine'), defaultName);
    if (!name) return;

    const savedKey = 'indexeddb://' + name;
    if (!listed[savedKey]) {
      alert('Kein gespeichertes Modell unter dem Namen ' + name + '. Verfügbare Modelle: ' + (known.length ? known.join(', ') : 'keine'));
      return;
    }

    try {
      models[name] = await tf.loadLayersModel(savedKey);
      const modelKey = resolveSavedModelKey(name);
      if (modelKey) {
        renderLoadedModel(name, modelKey);
      } else {
        setStatus(`Modell geladen: ${name}. Alias-Typ unbekannt, bitte speichere clean/best/over oder nutze diesen Alias erneut.`);
      }
    } catch (error) {
      alert('Laden fehlgeschlagen: ' + error.message);
    }
  });

  document.getElementById('btn-export-model').addEventListener('click', async () => {
    const trainedNames = Object.keys(models);
    if (!trainedNames.length) {
      alert('Keine trainierten Modelle vorhanden. Bitte zuerst trainieren.');
      return;
    }

    const modelKey = prompt('Welches trainierte Modell exportieren? (clean, best, over)', trainedNames.includes('best') ? 'best' : trainedNames[0]);
    if (!modelKey) return;
    if (!models[modelKey]) {
      alert('Kein trainiertes Modell namens ' + modelKey + '. Verfügbare Modelle: ' + trainedNames.join(', '));
      return;
    }

    const alias = prompt('Exportiere dieses Modell als Alias-Datei:', modelKey);
    if (!alias) return;

    try {
      setStatus('Exportiere Modell...');
      const model = models[modelKey];

      const saveHandler = tf.io.withSaveHandler(async (data) => {
        try {
          const modelJson = {
            modelTopology: data.modelTopology,
            weightsManifest: [
              {
                paths: [alias + '.weights.bin'],
                weights: data.weightSpecs || []
              }
            ]
          };

          downloadFile(JSON.stringify(modelJson), alias + '.json', 'application/json');

          // Ensure we pass an ArrayBuffer for the binary download
          let weightData = data.weightData;
          if (weightData && weightData.buffer) weightData = weightData.buffer;

          // Stagger downloads slightly to avoid browser blocking multiple downloads
          setTimeout(() => {
            try {
              downloadFile(weightData, alias + '.weights.bin', 'application/octet-stream');
            } catch (dErr) {
              console.error('Binary download failed:', dErr);
            }
          }, 120);
          return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON', weightDataType: 'ArrayBuffer' } };
        } catch (err) {
          console.error('Download error:', err);
          throw err;
        }
      });

      await model.save(saveHandler);
      setStatus(`✓ Modell ${modelKey} exportiert: ${alias}.json und ${alias}.weights.bin`);
    } catch (error) {
      console.error('Export error:', error);
      alert('Export fehlgeschlagen: ' + error.message);
      setStatus('Export fehlgeschlagen.');
    }
  });

  document.getElementById('btn-import-model').addEventListener('click', async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.json,.bin';
    input.onchange = async (event) => {
      const files = Array.from(event.target.files || []);
      if (!files.length) {
        setStatus('Import abgebrochen: keine Dateien ausgewählt.');
        return;
      }

      const jsonFile = files.find((file) => file.name.endsWith('.json'));
      const binFile = files.find((file) => file.name.endsWith('.bin'));
      if (!jsonFile || !binFile) {
        alert('Bitte sowohl die JSON-Datei als auch die BIN-Datei auswählen.');
        setStatus('Import fehlgeschlagen: JSON- und BIN-Datei fehlen.');
        return;
      }

      setStatus('Importiere Modell...');
      try {
        const orderedFiles = [jsonFile, binFile];
        const loadedModel = await tf.loadLayersModel(tf.io.browserFiles(orderedFiles));
        const alias = jsonFile.name.replace(/\.json$/, '');
        models[alias] = loadedModel;

        let modelKey = resolveSavedModelKey(alias);
        if (!modelKey) {
          const chosenKey = prompt('Welcher Modelltyp ist dieses importierte Modell? (clean, best, over)', 'best');
          if (chosenKey && ['clean', 'best', 'over'].includes(chosenKey)) {
            modelKey = chosenKey;
            setSavedModelAlias(alias, modelKey);
          }
        }

        if (modelKey) {
          renderLoadedModel(alias, modelKey);
        } else {
          setStatus(`Modell importiert: ${alias}. Kein Modelltyp automatisch erkannt.`);
        }
      } catch (error) {
        alert('Import fehlgeschlagen: ' + error.message);
        setStatus('Import fehlgeschlagen: ' + error.message);
      }
    };
    input.click();
  });

  document.getElementById('btn-train-clean').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Clean-Modell...');
    const epochsClean = Number.parseInt(document.getElementById('epochsClean').value, 10) || DEFAULT_EPOCHS_CLEAN;
    await train('clean', dataset.xTrain, dataset.yTrain, epochsClean);
    const trainLoss = evaluate('clean', dataset.xTrain, dataset.yTrain);
    const testLoss = evaluate('clean', dataset.xTest, dataset.yTest);
    plotPredictionSingle('plot-clean-train', 'clean', dataset.xTrain, dataset.yTrain, 'Vorhersage unverrauscht — Train', 'train', false);
    plotPredictionSingle('plot-clean-test', 'clean', dataset.xTest, dataset.yTest, 'Vorhersage unverrauscht — Test', 'test', false);
    document.getElementById('loss-clean-train').textContent = `Train MSE: ${trainLoss.toExponential(3)}`;
    document.getElementById('loss-clean-test').textContent = `Test MSE: ${testLoss.toExponential(3)}`;
    setStatus(`Clean-Modell trainiert (${epochsClean} Epochen).`);
  });

  document.getElementById('btn-train-best').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Best-Fit-Modell...');
    await train('best', dataset.xTrain, dataset.yTrainNoisy, parseInt(document.getElementById('epochsBest').value, 10));
    const trainLoss = evaluate('best', dataset.xTrain, dataset.yTrainNoisy);
    const testLoss = evaluate('best', dataset.xTest, dataset.yTestNoisy);
    plotPredictionSingle('plot-best-train', 'best', dataset.xTrain, dataset.yTrainNoisy, 'Best-Fit — Train (verrauscht)', 'train', true);
    plotPredictionSingle('plot-best-test', 'best', dataset.xTest, dataset.yTestNoisy, 'Best-Fit — Test (verrauscht)', 'test', true);
    document.getElementById('loss-best-train').textContent = `Train MSE: ${trainLoss.toExponential(3)}`;
    document.getElementById('loss-best-test').textContent = `Test MSE: ${testLoss.toExponential(3)}`;
    setStatus('Best-Fit-Modell trainiert.');
  });

  document.getElementById('btn-train-over').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Overfit-Modell...');
    await train('over', dataset.xTrain, dataset.yTrainNoisy, parseInt(document.getElementById('epochsOver').value, 10));
    const trainLoss = evaluate('over', dataset.xTrain, dataset.yTrainNoisy);
    const testLoss = evaluate('over', dataset.xTest, dataset.yTestNoisy);
    plotPredictionSingle('plot-over-train', 'over', dataset.xTrain, dataset.yTrainNoisy, 'Overfit — Train (verrauscht)', 'train', true);
    plotPredictionSingle('plot-over-test', 'over', dataset.xTest, dataset.yTestNoisy, 'Overfit — Test (verrauscht)', 'test', true);
    document.getElementById('loss-over-train').textContent = `Train MSE: ${trainLoss.toExponential(3)}`;
    document.getElementById('loss-over-test').textContent = `Test MSE: ${testLoss.toExponential(3)}`;
    setStatus('Overfit-Modell trainiert.');
  });
}

window.addEventListener('DOMContentLoaded', () => {
  setupEventHandlers();
  initialTraining();
});
