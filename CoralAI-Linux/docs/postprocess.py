# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process libcoral files from Sphinx to optimize the HTML for Coral website."""

import argparse
import os
import re

from bs4 import BeautifulSoup


def remove_title(soup):
  """Deletes the extra H1 title."""
  h1 = soup.find('h1')
  if h1:
    h1.extract()
  return soup


def relocate_h2id(soup):
  """Moves the anchor ID to the H2 tag, from the wrapper DIV."""
  for h2 in soup.find_all('h2'):
    div = h2.find_parent('div')
    if div.has_attr('id') and not h2.has_attr('id'):
      # print('Move ID: ' + div['id'])
      h2['id'] = div['id']
      del div['id']
    # Also delete embedded <a> tag
    if h2.find('a'):
      h2.find('a').extract()
  return soup


def clean_pre(soup):
  """Adds our prettyprint class to PRE and removes some troubelsome tags."""
  for pre in soup.find_all('pre'):
    pre['class'] = 'language-cpp'
    # This effectively deletes the wrapper DIV and P tags that cause issues
    parent_p = pre.find_parent('p')
    if parent_p:
      parent_p.replace_with(pre)
  return soup


def remove_coral(soup):
  """Removes 'coral' namespace link that does nothing."""
  for a in soup.select('a[title=coral]'):
    content = a.contents[0]
    a.replace_with(content)
  return soup


def remove_subclass_string(soup):
  """Removes description of internal subclass names."""
  paras = soup.find_all('p', string=re.compile(r'^Subclassed by'))
  for p in paras:
    p.extract()
  return soup


def style_source_link(soup):
  """Adds class to allow styling of the source links."""
  paras = soup.find_all('p', string=re.compile(r'source]$'))
  for p in paras:
    p['class'] = 'cpp-source-link'
  return soup


def clean_index(soup):
  """Removes the HTML table from the index page, leaving just lists of APIs."""
  uls = soup.find_all('ul')
  for child in soup.find('div', class_='sphinx-reference').children:
    child.extract()
  for ul in uls:
    soup.find('div', class_='sphinx-reference').append(ul)
  return soup


def process(file):
  """Runs all the cleanup functions."""
  print('Post-processing ' + file)
  soup = BeautifulSoup(open(file), 'html.parser')
  soup = remove_title(soup)
  soup = relocate_h2id(soup)
  soup = clean_pre(soup)
  soup = remove_coral(soup)
  soup = remove_subclass_string(soup)
  soup = style_source_link(soup)
  if os.path.split(file)[1] == 'index.md':
    soup = clean_index(soup)
  with open(file, 'w') as output:
    output.write(str(soup))


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-f', '--file', required=True, help='File path of HTML file(s).')
  args = parser.parse_args()

  # Accept a directory or single file
  if os.path.isdir(args.file):
    for file in os.listdir(args.file):
      if os.path.splitext(file)[1] == '.md':
        process(os.path.join(args.file, file))
  else:
    process(args.file)


if __name__ == '__main__':
  main()
