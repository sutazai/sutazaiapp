"""
This module uses ctypes to bind a whole bunch of functions and constants from
SecureTransport. The goal here is to provide the low-level API to
SecureTransport. These are essentially the C-level functions and constants, and
they're pretty gross to work with.

This code is a bastardised version of the code found in Will Bond's oscrypto
library. An enormous debt is owed to him for blazing this trail for us. For
that reason, this code should be considered to be covered both by urllib3's
license and by oscrypto's:

    Copyright (c) 2015-2016 Will Bond <will@wbond.net>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

from __future__ import absolute_import

import platform
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    c_bool,
    c_byte,
    c_char_p,
    c_int32,
    c_long,
    c_size_t,
    c_uint32,
    c_ulong,
    c_void_p,
)
from ctypes.util import find_library

from ...packages.six import raise_from

if platform.system() != "Darwin":
    raise ImportError("Only macOS is supported")

version = platform.mac_ver()[0]
version_info = tuple(map(int, version.split(".")))
if version_info < (10, 8):
    raise OSError(
        "Only OS X 10.8 and newer are supported, not %s.%s"
        % (version_info[0], version_info[1])
    )


def load_cdll(name, macos10_16_path):
    """Loads a CDLL by name, falling back to known path on 10.16+"""
    try:
        # Big Sur is technically 11 but we use 10.16 due to the Big Sur
        # beta being labeled as 10.16.
        if version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError  # Caught and reraised as 'ImportError'
        return CDLL(path, use_errno=True)
    except OSError:
        raise_from(ImportError("The library %s failed to load" % name), None)


)
CoreFoundation = load_cdll(
    "CoreFoundation",
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
)


Boolean = c_bool
CFIndex = c_long
CFStringEncoding = c_uint32
CFData = c_void_p
CFString = c_void_p
CFArray = c_void_p
CFMutableArray = c_void_p
CFDictionary = c_void_p
CFError = c_void_p
CFType = c_void_p
CFTypeID = c_ulong

CFTypeRef = POINTER(CFType)
CFAllocatorRef = c_void_p

OSStatus = c_int32

CFDataRef = POINTER(CFData)
CFStringRef = POINTER(CFString)
CFArrayRef = POINTER(CFArray)
CFMutableArrayRef = POINTER(CFMutableArray)
CFDictionaryRef = POINTER(CFDictionary)
CFArrayCallBacks = c_void_p
CFDictionaryKeyCallBacks = c_void_p
CFDictionaryValueCallBacks = c_void_p

SecCertificateRef = POINTER(c_void_p)
SecExternalFormat = c_uint32
SecExternalItemType = c_uint32
SecIdentityRef = POINTER(c_void_p)
SecItemImportExportFlags = c_uint32
SecItemImportExportKeyParameters = c_void_p
SecKeychainRef = POINTER(c_void_p)
SSLProtocol = c_uint32
SSLCipherSuite = c_uint32
SSLContextRef = POINTER(c_void_p)
SecTrustRef = POINTER(c_void_p)
SSLConnectionRef = c_uint32
SecTrustResultType = c_uint32
SecTrustOptionFlags = c_uint32
SSLProtocolSide = c_uint32
SSLConnectionType = c_uint32
SSLSessionOption = c_uint32


try:
        CFDataRef,
        CFStringRef,
        POINTER(SecExternalFormat),
        POINTER(SecExternalItemType),
        SecItemImportExportFlags,
        POINTER(SecItemImportExportKeyParameters),
        SecKeychainRef,
        POINTER(CFArrayRef),
    ]




        CFAllocatorRef,
        CFDataRef,
    ]



        CFTypeRef,
        SecCertificateRef,
        POINTER(SecIdentityRef),
    ]

        c_char_p,
        c_uint32,
        c_void_p,
        Boolean,
        c_void_p,
        POINTER(SecKeychainRef),
    ]


        CFDataRef,
        CFDictionaryRef,
        POINTER(CFArrayRef),
    ]

    SSLReadFunc = CFUNCTYPE(
        OSStatus, SSLConnectionRef, c_void_p, POINTER(c_size_t)
    )
    SSLWriteFunc = CFUNCTYPE(
        OSStatus, SSLConnectionRef, POINTER(c_byte), POINTER(c_size_t)
    )

        SSLContextRef,
        SSLReadFunc,
        SSLWriteFunc,
    ]



        SSLContextRef,
        CFTypeRef,
        Boolean,
    ]


        SSLContextRef,
        c_char_p,
        c_size_t,
    ]


        SSLContextRef,
        c_char_p,
        c_size_t,
        POINTER(c_size_t),
    ]

        SSLContextRef,
        c_char_p,
        c_size_t,
        POINTER(c_size_t),
    ]


        SSLContextRef,
        POINTER(c_size_t),
    ]

        SSLContextRef,
        POINTER(SSLCipherSuite),
        POINTER(c_size_t),
    ]

        SSLContextRef,
        POINTER(SSLCipherSuite),
        c_size_t,
    ]

        SSLContextRef,
        POINTER(c_size_t),
    ]

        SSLContextRef,
        POINTER(SSLCipherSuite),
        POINTER(c_size_t),
    ]

        SSLContextRef,
        POINTER(SSLCipherSuite),
    ]

        SSLContextRef,
        POINTER(SSLProtocol),
    ]



        SecTrustRef,
        Boolean,
    ]

        SecTrustRef,
        POINTER(SecTrustResultType),
    ]



        CFAllocatorRef,
        SSLProtocolSide,
        SSLConnectionType,
    ]

        SSLContextRef,
        SSLSessionOption,
        Boolean,
    ]



    try:
    except AttributeError:
        # Supported only in 10.12+
        pass



    )
    )

    # CoreFoundation time!
    CoreFoundation.CFRetain.argtypes = [CFTypeRef]
    CoreFoundation.CFRetain.restype = CFTypeRef

    CoreFoundation.CFRelease.argtypes = [CFTypeRef]
    CoreFoundation.CFRelease.restype = None

    CoreFoundation.CFGetTypeID.argtypes = [CFTypeRef]
    CoreFoundation.CFGetTypeID.restype = CFTypeID

    CoreFoundation.CFStringCreateWithCString.argtypes = [
        CFAllocatorRef,
        c_char_p,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringCreateWithCString.restype = CFStringRef

    CoreFoundation.CFStringGetCStringPtr.argtypes = [
        CFStringRef,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringGetCStringPtr.restype = c_char_p

    CoreFoundation.CFStringGetCString.argtypes = [
        CFStringRef,
        c_char_p,
        CFIndex,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringGetCString.restype = c_bool

    CoreFoundation.CFDataCreate.argtypes = [CFAllocatorRef, c_char_p, CFIndex]
    CoreFoundation.CFDataCreate.restype = CFDataRef

    CoreFoundation.CFDataGetLength.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetLength.restype = CFIndex

    CoreFoundation.CFDataGetBytePtr.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetBytePtr.restype = c_void_p

    CoreFoundation.CFDictionaryCreate.argtypes = [
        CFAllocatorRef,
        POINTER(CFTypeRef),
        POINTER(CFTypeRef),
        CFIndex,
        CFDictionaryKeyCallBacks,
        CFDictionaryValueCallBacks,
    ]
    CoreFoundation.CFDictionaryCreate.restype = CFDictionaryRef

    CoreFoundation.CFDictionaryGetValue.argtypes = [CFDictionaryRef, CFTypeRef]
    CoreFoundation.CFDictionaryGetValue.restype = CFTypeRef

    CoreFoundation.CFArrayCreate.argtypes = [
        CFAllocatorRef,
        POINTER(CFTypeRef),
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreate.restype = CFArrayRef

    CoreFoundation.CFArrayCreateMutable.argtypes = [
        CFAllocatorRef,
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreateMutable.restype = CFMutableArrayRef

    CoreFoundation.CFArrayAppendValue.argtypes = [CFMutableArrayRef, c_void_p]
    CoreFoundation.CFArrayAppendValue.restype = None

    CoreFoundation.CFArrayGetCount.argtypes = [CFArrayRef]
    CoreFoundation.CFArrayGetCount.restype = CFIndex

    CoreFoundation.CFArrayGetValueAtIndex.argtypes = [CFArrayRef, CFIndex]
    CoreFoundation.CFArrayGetValueAtIndex.restype = c_void_p

    CoreFoundation.kCFAllocatorDefault = CFAllocatorRef.in_dll(
        CoreFoundation, "kCFAllocatorDefault"
    )
    CoreFoundation.kCFTypeArrayCallBacks = c_void_p.in_dll(
        CoreFoundation, "kCFTypeArrayCallBacks"
    )
    CoreFoundation.kCFTypeDictionaryKeyCallBacks = c_void_p.in_dll(
        CoreFoundation, "kCFTypeDictionaryKeyCallBacks"
    )
    CoreFoundation.kCFTypeDictionaryValueCallBacks = c_void_p.in_dll(
        CoreFoundation, "kCFTypeDictionaryValueCallBacks"
    )

    CoreFoundation.CFTypeRef = CFTypeRef
    CoreFoundation.CFArrayRef = CFArrayRef
    CoreFoundation.CFStringRef = CFStringRef
    CoreFoundation.CFDictionaryRef = CFDictionaryRef

except AttributeError:
    raise ImportError("Error initializing ctypes")


class CFConst(object):
    """
    A class object that acts as essentially a namespace for CoreFoundation
    constants.
    """

    kCFStringEncodingUTF8 = CFStringEncoding(0x08000100)


    """
    """

    kSSLSessionOptionBreakOnServerAuth = 0

    kSSLProtocol2 = 1
    kSSLProtocol3 = 2
    kTLSProtocol1 = 4
    kTLSProtocol11 = 7
    kTLSProtocol12 = 8
    # SecureTransport does not support TLS 1.3 even if there's a constant for it
    kTLSProtocol13 = 10
    kTLSProtocolMaxSupported = 999

    kSSLClientSide = 1
    kSSLStreamType = 0

    kSecFormatPEMSequence = 10

    kSecTrustResultInvalid = 0
    kSecTrustResultProceed = 1
    # This gap is present on purpose: this was kSecTrustResultConfirm, which
    # is deprecated.
    kSecTrustResultDeny = 3
    kSecTrustResultUnspecified = 4
    kSecTrustResultRecoverableTrustFailure = 5
    kSecTrustResultFatalTrustFailure = 6
    kSecTrustResultOtherError = 7

    errSSLProtocol = -9800
    errSSLWouldBlock = -9803
    errSSLClosedGraceful = -9805
    errSSLClosedNoNotify = -9816
    errSSLClosedAbort = -9806

    errSSLXCertChainInvalid = -9807
    errSSLCrypto = -9809
    errSSLInternal = -9810
    errSSLCertExpired = -9814
    errSSLCertNotYetValid = -9815
    errSSLUnknownRootCert = -9812
    errSSLNoRootCert = -9813
    errSSLHostNameMismatch = -9843
    errSSLPeerHandshakeFail = -9824
    errSSLPeerUserCancelled = -9839
    errSSLWeakPeerEphemeralDHKey = -9850
    errSSLServerAuthCompleted = -9841
    errSSLRecordOverflow = -9847

    errSecVerifyFailed = -67808
    errSecNoTrustSettings = -25263
    errSecItemNotFound = -25300
    errSecInvalidTrustSettings = -25262

    # Cipher suites. We only pick the ones our default cipher string allows.
    TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 = 0xC02C
    TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 = 0xC030
    TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 = 0xC02B
    TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 = 0xC02F
    TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 = 0xCCA9
    TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 = 0xCCA8
    TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 = 0x009F
    TLS_DHE_RSA_WITH_AES_128_GCM_SHA256 = 0x009E
    TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384 = 0xC024
    TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384 = 0xC028
    TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA = 0xC00A
    TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA = 0xC014
    TLS_DHE_RSA_WITH_AES_256_CBC_SHA256 = 0x006B
    TLS_DHE_RSA_WITH_AES_256_CBC_SHA = 0x0039
    TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256 = 0xC023
    TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256 = 0xC027
    TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA = 0xC009
    TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA = 0xC013
    TLS_DHE_RSA_WITH_AES_128_CBC_SHA256 = 0x0067
    TLS_DHE_RSA_WITH_AES_128_CBC_SHA = 0x0033
    TLS_RSA_WITH_AES_256_GCM_SHA384 = 0x009D
    TLS_RSA_WITH_AES_128_GCM_SHA256 = 0x009C
    TLS_RSA_WITH_AES_256_CBC_SHA256 = 0x003D
    TLS_RSA_WITH_AES_128_CBC_SHA256 = 0x003C
    TLS_RSA_WITH_AES_256_CBC_SHA = 0x0035
    TLS_RSA_WITH_AES_128_CBC_SHA = 0x002F
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_AES_128_CCM_8_SHA256 = 0x1305
    TLS_AES_128_CCM_SHA256 = 0x1304
