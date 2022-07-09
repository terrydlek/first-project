class car:
    def __init__(self, speed, max_speed, weight, name, color):
        self.speed = speed
        self.max_speed = max_speed
        self.weight = weight
        self.name = name
        self.color = color
    def upSpeed(self, speed):
        self.after_speed = 0
        if speed >= self.max_speed:
            self.after_speed = self.max_speed
        else:
            self.after_speed = speed
        return self.after_speed

    def downSpeed(self, speed):
        self.afterDown_speed = 0
        if speed >= self.after_speed:
            self.afterDown_speed = 0
        else:
            self.afterDown_speed = self.after_speed - speed
        return self.afterDown_speed

class truck(car):
    def __init__(self, name, color, speed=0, max_speed=100, weight=5000, load=0, max_load=2000):
        super().__init__(speed, max_speed, weight, name, color)
        self.type = "트럭"
        self.load = load
        self.max_load = max_load

    def upSpeed(self, speed):
        super().upSpeed(speed)
        return print("%s의 가속 후 속도: %d" % (self.name, self.after_speed))

    def downSpeed(self, speed):
        super().downSpeed(speed)
        return print("%s의 가속 후 속도: %d" % (self.name, self.afterDown_speed))

    def upLoad(self, load):
        self.afterup_load = 0
        if load >= self.max_load:
            self.afterup_load = self.max_load
        else:
            self.afterup_load = load
        return print("%s의 상차 후 적재량: %d" % (self.name, self.afterup_load))

    def downLoad(self, load):
        self.afterdown_load = 0
        if load >= self.afterup_load:
            self.afterdown_load = 0
        else:
            self.afterdown_load = self.afterup_load - load
        return print("%s의 하차 후 적재량: %d" % (self.name, self.afterdown_load))

    def __str__(self):
        print("@이름:%s" % (self.name))
        return "-분류:%s, 중량:%d, 최대속도:%d, 최대 적재량:%d" % (self.type, self.weight, self.max_speed, self.max_load)

class sedan(car):
    def __init__(self, name, color, speed=0, max_speed=250, weight=2000, num_seat=5):
        super().__init__(speed, max_speed, weight, name, color)
        self.type = "승용차"
        self.num_seat = num_seat

    def upSpeed(self, speed):
        super().upSpeed(speed)
        return print("%s의 가속 후 속도: %d" % (self.name, self.after_speed))

    def downSpeed(self, speed):
        super().downSpeed(speed)
        return print("%s의 가속 후 속도: %d" % (self.name, self.afterDown_speed))

    def __str__(self):
        print("@이름:%s" % (self.name))
        return "-분류:%s, 중량:%d, 최대속도:%d, 좌석 수:%d" % (self.type, self.weight, self.max_speed, self.num_seat)


car1 = truck('봉고', '파랑')
car2 = sedan('소나타', '검정')
print(car1)
print(car2)
print()

car1.upSpeed(300)
car2.upSpeed(300)
print()

car1.downSpeed(200)
car2.downSpeed(200)
print()

car1.upLoad(3000)
car1.downLoad(1000)
