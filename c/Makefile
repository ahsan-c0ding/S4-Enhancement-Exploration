CC = gcc
CFLAGS = -Wall -Wextra -O2
all: galaxy_app test_app

galaxy_app: main.c nn.c math.c
	$(CC) $(CFLAGS) -o galaxy_app main.c nn.c math.c

test_app: test.c nn.c math.c
	$(CC) $(CFLAGS) -o test_app test.c nn.c math.c

clean:
	rm -f galaxy_app test_app