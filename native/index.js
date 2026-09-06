'use strict';
const path = require('path');
const native = require(path.join(__dirname, 'build', 'Release', 'pikelet_native.node'));
module.exports = native;
