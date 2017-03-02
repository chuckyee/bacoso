import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, \
    UnconvergedErrorHandler, AliasingErrorHandler, FrozenJobErrorHandler, \
    PositiveEnergyErrorHandler, MeshSymmetryErrorHandler
from custodian.vasp.jobs import VaspJob


handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
            AliasingErrorHandler(), FrozenJobErrorHandler(),
            PositiveEnergyErrorHandler(), MeshSymmetryErrorHandler()]
jobs = VaspJob(sys.argv[1:])
c = Custodian(handlers, jobs, max_errors=10)
c.run()
