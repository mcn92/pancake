'use strict';
const path = require('path');
const native = require(path.join(__dirname, 'build', 'Release', 'pancake_native.node'));
module.exports = native;
