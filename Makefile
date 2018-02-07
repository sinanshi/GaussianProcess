all:
	python3 setup.py install

build:
	python3 setup.py build

clean:
	rm -rf gp_emulator.egg-info dist build
