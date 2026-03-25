importScripts('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');

let worker = null;

async function initWorker() {
    worker = await Tesseract.createWorker('eng');
    await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        tessedit_pageseg_mode: '7',  // single line
    });
}

const ready = initWorker();

self.onmessage = async (event) => {
    const { cardId, cropData, cropWidth, cropHeight } = event.data;
    try {
        await ready;

        // Expand grayscale to RGBA for Tesseract
        const rgba = new Uint8ClampedArray(cropWidth * cropHeight * 4);
        for (let i = 0; i < cropData.length; i++) {
            const v = cropData[i];
            rgba[i * 4]     = v;
            rgba[i * 4 + 1] = v;
            rgba[i * 4 + 2] = v;
            rgba[i * 4 + 3] = 255;
        }

        const canvas = new OffscreenCanvas(cropWidth, cropHeight);
        const ctx = canvas.getContext('2d');
        ctx.putImageData(new ImageData(rgba, cropWidth, cropHeight), 0, 0);
        const blob = await canvas.convertToBlob({ type: 'image/png' });

        const { data: { text } } = await worker.recognize(blob);

        self.postMessage({ cardId, text: text.trim() });
    } catch (err) {
        self.postMessage({ cardId, text: '', error: err.message });
    }
};
