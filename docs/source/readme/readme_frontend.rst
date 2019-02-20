***************************
Nirdizati Research frontend
***************************

|License| |HitCount|

Master

|Build Status master| |Coverage Status master| |Total alerts master| |Language grade: Javascript master|


Development

|Build Status development| |Coverage Status development|

React frontend to perform Predictive Monitoring analysis over event logs.

Running in a new environment
============================
The docker build is available @ https://hub.docker.com/r/nirdizatiresearch/predict-react/ in any case if you prefer to setup your environment on yuor own you can refer the `Dockerfile <Dockerfile>`_.

Docker Compose
==============
To run the project:

.. code::

    docker-compose up react-client


Run an instance of the project
==============================
If you are familiar with docker-compose the `docker-compose <docker-compose.yml>`_ file is available, otherwise if you use pycharm as IDE is available the run configuration in the `runConfiguration <.idea/runConfiguration>`_ settings.

package.json contains all supported commands for this project.

Install required components:

.. code::

    npm install


Run build:

.. code::

    npm run build


Run tests:

.. code::

    npm run test


Run start:

.. code::

    npm run start


Thanks to
=========
This project was bootstrapped with `Create React App <https://github.com/facebookincubator/create-react-app>`_ and `Storybook <https://github.com/storybooks/storybook>`_.

Contributors
============
-  `@stebranchi <https://github.com/stebranchi>`_ Stefano Branchi
-  `@dfmchiara <https://github.com/dfmchiara>`_ Chiara Di Francescomarino
-  `@TKasekamp <https://github.com/TKasekamp>`_ Tõnis Kasekamp
-  `@mrsonuk <https://github.com/mrsonuk>`_ Santosh Kumar
-  `@fmmaggi <https://github.com/fmmaggi>`_ Fabrizio Maggi
-  `@WilliamsRizzi <https://github.com/WilliamsRizzi>`_ Williams Rizzi
-  `@HitLuca <https://github.com/HitLuca>`_ Luca Simonetto


.. |Build Status master| image:: https://travis-ci.org/nirdizati-research/predict-react.svg?branch=master
   :target: https://travis-ci.org/nirdizati-research/predict-react

.. |Coverage Status master| image:: https://codecov.io/gh/nirdizati-research/predict-react/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nirdizati-research/predict-react

.. |Total alerts master| image:: https://img.shields.io/lgtm/alerts/g/nirdizati-research/predict-react.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/nirdizati-research/predict-react/alerts/

.. |Language grade: Javascript master| image:: https://img.shields.io/lgtm/grade/javascript/g/nirdizati-research/predict-react.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/nirdizati-research/predict-react/context:javascript

.. |Build Status development| image:: https://travis-ci.org/nirdizati-research/predict-react.svg?branch=development
   :target: https://travis-ci.org/nirdizati-research/predict-react

.. |Coverage Status development| image:: https://codecov.io/gh/nirdizati-research/predict-react/branch/development/graph/badge.svg
   :target: https://codecov.io/gh/nirdizati-research/predict-react

.. |HitCount| image:: http://hits.dwyl.io/nirdizati-research/predict-react.svg
   :target: http://hits.dwyl.io/nirdizati-research/predict-react

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
