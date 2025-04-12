#define BOOST_TEST_MODULE AddNumbersTest
#include <boost/test/included/unit_test.hpp>
#include "adaptiveSubtraction.cpp"

BOOST_AUTO_TEST_CASE(AddNumbersTest) 
{
    BOOST_CHECK_EQUAL(add(2, 3), 5);
    BOOST_CHECK_EQUAL(add(-1, 1), 0); 
    BOOST_CHECK_EQUAL(add(0, 0), 0); 
    BOOST_CHECK_EQUAL(add(-5, -5), -10); 
}
