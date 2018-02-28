all:
	python setup.py install

build:
	python setup.py build

clean:
	rm -rf gp_emulator.egg-info dist build
