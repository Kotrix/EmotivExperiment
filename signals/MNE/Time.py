class Time:

    @staticmethod
    def to_sample(time):
        frequency = 128
        return int(round(time * frequency))

