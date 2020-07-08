

all: build

build:
	python setup.py build

install:
	python setup.py install

clean:
	rm -rf build

check:
	python test/test.py
