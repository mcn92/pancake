import { serveFromR2, onOptions } from '../_serve-r2.js';

export function onRequestGet(context) {
    return serveFromR2(context, 'models');
}

export function onRequestOptions() {
    return onOptions();
}
