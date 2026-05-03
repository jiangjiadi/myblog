'use strict';

const katexCss = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css">';

hexo.extend.injector.register('head_end', katexCss, 'post');
hexo.extend.injector.register('head_end', katexCss, 'page');
