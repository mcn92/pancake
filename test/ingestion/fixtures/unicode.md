Guía de la API
==============

Los encabezados setext con signo igual son títulos de primer nivel y deben
tratarse exactamente como un encabezado ATX de nivel uno, incluido el
tratamiento del título del documento y su exclusión de las rutas.

Über uns
--------

Setext headings underlined with dashes are level two, and this one carries
a non-ASCII character that the slug must preserve in lowercase form, since
that is what the rendered anchor id actually contains on the page.

## 日本語の見出し

A heading written entirely in Japanese keeps every character in its anchor
because the slug alphabet is Unicode letters and numbers, not ASCII, and
the section text below it must remain attached to exactly this heading.

- not a heading
--------------------

The dashed line above follows a list item, which cannot be promoted into a
setext heading; it must be treated as plain content of this section, and
this paragraph stays inside the Japanese section where it started.
