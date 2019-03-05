import math

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, rhs):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar):
        return Vec3(scalar * self.x, scalar * self.y, scalar * self.z)

    def __add__(self, rhs):
        return Vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

    def __sub__(self, rhs):
        return Vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)

    def normalize(self):
        length = math.sqrt(self.dot(self))
        return Vec3(self.x / length, self.y / length, self.z / length)

    def __str__(self):
        return "Vector3({0:3f},{1:3f},{2:3f})".format(self.x, self.y, self.z)

    def __repr__(self):
        return "Vector3({0:3f},{1:3f},{2:3f})".format(self.x, self.y, self.z)


class Quat:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def dot(self, rhs):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w

    def __neg__(self):
        return Quat(-self.x, -self.y, -self.z, -self.w)

    def normalize(self):
        length = math.sqrt(self.dot(self))
        return Quat(self.x / length, self.y / length, self.z / length, self.w / length)

    def angleAxis(self):
        q = self.normalize()
        angle = 2.0 * math.acos(q.w)
        s = 1.0 - (q.w * q.w)
        if s <= 0.0:
            return angle, Vec3(1, 0, 0)
        else:
            sqrts = math.sqrt(s)
            return angle, Vec3(q.x / sqrts, q.y / sqrts, q.z / sqrts)

    def log(self):
        cosHalfAngle = self.w

        if cosHalfAngle > 1.0:
            cosHalfAngle = 1.0
        if cosHalfAngle < -1.0:
            cosHalfAngle = -1.0

        sinHalfAngle = math.sqrt(1.0 - cosHalfAngle * cosHalfAngle)

        if abs(sinHalfAngle) < 0.0005:
            sinHalfAngle = 1

        angle = 2.0 * math.acos(cosHalfAngle)
        return Quat((self.x / sinHalfAngle) * angle,
                    (self.y / sinHalfAngle) * angle,
                    (self.z / sinHalfAngle) * angle, 0)

    def conjugate(self):
        return Quat(-self.x, -self.y, -self.z, self.w)

    def inverse(self):
        return self.conjugate()

    def __mul__(self, rhs):
        return Quat( self.x * rhs.w + self.y * rhs.z - self.z * rhs.y + self.w * rhs.x,
                    -self.x * rhs.z + self.y * rhs.w + self.z * rhs.x + self.w * rhs.y,
                     self.x * rhs.y - self.y * rhs.x + self.z * rhs.w + self.w * rhs.z,
                    -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z + self.w * rhs.w);

    def exp(self):
        angle = math.sqrt(self.dot(self))
        if angle > 0.0001:
            sinHalfAngle = math.sin(0.5 * angle)
            cosHalfAngle = math.cos(0.5 * angle)
            return Quat((self.x / angle) * sinHalfAngle,
                        (self.y / angle) * sinHalfAngle,
                        (self.z / angle) * sinHalfAngle,
                        cosHalfAngle)
        else:
            return Quat(0, 0, 0, 1)

    def rotate(self, rhs):
        result = self * Quat(rhs.x, rhs.y, rhs.z, 0.0) * self.conjugate()
        return Vec3(result.x, result.y, result.z)

    def swingTwistDecomposition(self, referenceDirection):
        direction = referenceDirection.normalize()
        axisOfRotation = Vec3(self.x, self.y, self.z)
        twistImaginaryPart = direction * direction.dot(axisOfRotation)
        twist = Quat(twistImaginaryPart.x, twistImaginaryPart.y, twistImaginaryPart.z, self.w).normalize()
        swing = self * twist.inverse()
        return swing, twist

    def __str__(self):
        return "Quat({0:.3f},{1:.3f},{2:.3f},{3:.3f})".format(self.x, self.y, self.z, self.w)

    def __repr__(self):
        return "Quat({0:.3f},{1:.3f},{2:.3f},{3:.3f})".format(self.x, self.y, self.z, self.w)


class Xform:
    def __init__(self, pos, rot):
        self.pos = pos
        self.rot = rot

    def __mul__(self, rhs):
        rot = self.rot * rhs.rot
        pos = self.pos + self.rot.rotate(rhs.pos)
        return Xform(pos, rot)

    def inverse(self):
        invRot = self.rot.inverse()
        invPos = -self.pos
        return Xform(invRot.rotate(invPos), invRot)

    def xformVector(self, vector):
        return self.rot.rotate(vector)

    def xformPoint(self, point):
        return self.rot.rotate(point) + self.pos

    def __str__(self):
        return "Xform(pos=[{0:.3f},{1:.3f},{2:.3f}],rot=[{3:.3f},{4:.3f},{5:.3f},{6:.3f}])".format(self.pos.x, self.pos.y, self.pos.z, self.rot.x, self.rot.y, self.rot.z, self.rot.w)

    def __repr__(self):
        return "Xform(pos=[{0:.3f},{1:.3f},{2:.3f}],rot=[{3:.3f},{4:.3f},{5:.3f},{6:.3f}])".format(self.pos.x, self.pos.y, self.pos.z, self.rot.x, self.rot.y, self.rot.z, self.rot.w)
