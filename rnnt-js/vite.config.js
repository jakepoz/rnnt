// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: [
    'models/**/*.bin',
  ],

  // Needed to add according to instructions here: https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
  },
});